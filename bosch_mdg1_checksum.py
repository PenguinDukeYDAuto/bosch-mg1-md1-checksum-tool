"""
Bosch MDG1 TC2xx ECU checksum 工具
build by dukeapenguin

适用 MG1CS00x / MG1CS04x / MD1CSxxx / MDG1_MD1CPxxx / RBA1DGS1/* 8MB BIN

用法:
    python bosch_mdg1_checksum.py firmware.bin                # 校验
    python bosch_mdg1_checksum.py firmware.bin -w fixed.bin   # 重算并写回

DLL 函数对应:
    FUN_10001070/1110/1140  -> _crc32       多项式 0xEDB88320
    FUN_10001180            -> _bsum
    FUN_10002bb0            -> _add32
    FUN_10002c00            -> _add16
    FUN_10001880            -> find_slots
    FUN_10002fe0/10002c50   -> _parse_sub_regions
    FUN_10003280            -> compute_algo
    FUN_10003a10            -> verify / recompute
    FUN_10001680            -> validate_trailer
"""

import struct
import zlib
from dataclasses import dataclass, field
from typing import Iterator, List, Optional


# ====== 常量 ======

# 算法 ID（FUN_10003280 switch case）
ALGO_ADD32 = 0x01
ALGO_CRC32 = 0x02   # 0x82 同
ALGO_BSUM  = 0x08
ALGO_ADD16 = 0x10
ALGO_WRMG  = 0xFF   # 写魔数（0xAA55CC33 翻转）

ALGO_NAMES = {0x01: "ADD32", 0x02: "CRC32", 0x82: "CRC32",
              0x08: "BSUM",  0x10: "ADD16", 0xFF: "WRMG"}

# 字节序 magic（FUN_100012d0）
MAGIC_LE = 0xA01F
MAGIC_BE = 0x5777

# TC2xx PFlash 地址段
TC2XX_PFLASH_BASE = 0x80000000
TC2XX_PFLASH_END  = 0x807FFFFF


# ====== 基础工具 ======

def bswap16(x: int) -> int:
    return ((x & 0xFF) << 8) | ((x >> 8) & 0xFF)


def bswap32(x: int) -> int:
    return struct.unpack('<I', struct.pack('>I', x & 0xFFFFFFFF))[0]


def addr_to_offset(addr: int, base: int = TC2XX_PFLASH_BASE) -> Optional[int]:
    # TC2xx 绝对地址转 BIN 文件偏移
    if base <= addr <= base + 0x7FFFFF:
        return addr - base
    return None


def _crc32(data: bytes, start: int, end_incl: int) -> int:
    # zlib.crc32 = 标准 CRC-32 (poly 0xEDB88320, init/final 0xFFFFFFFF)
    # FUN_10001140 的循环是 off<=end，end 包含
    return zlib.crc32(data[start:end_incl + 1]) & 0xFFFFFFFF


def _bsum(data: bytes, start: int, end_excl: int, init: int = 0xFFFFFFFF) -> int:
    # FUN_10001180: pos<end，end 不含
    return (init + sum(data[start:end_excl])) & 0xFFFFFFFF


def _add32(data: bytes, start: int, end_excl: int, init: int = 0xFFFFFFFF) -> int:
    # FUN_10002bb0: 累加 LE u32，pos<end
    s = init
    pos = start
    while pos + 3 < end_excl:
        s = (s + struct.unpack_from('<I', data, pos)[0]) & 0xFFFFFFFF
        pos += 4
    return s


def _add16(data: bytes, start: int, end_excl: int, init: int = 0xFFFFFFFF) -> int:
    # FUN_10002c00: 累加 LE u16，pos<end
    s = init
    pos = start
    while pos + 1 < end_excl:
        s = (s + (struct.unpack_from('<H', data, pos)[0] & 0xFFFF)) & 0xFFFFFFFF
        pos += 2
    return s


def compute_algo(algo: int, data: bytes, start: int, end_incl: int) -> Optional[int]:
    # algo 字节分发，未知算法返回 None
    if algo in (0x02, 0x82):
        return _crc32(data, start, end_incl)
    if algo == 0x01:
        return _add32(data, start, end_incl)
    if algo == 0x10:
        return _add16(data, start, end_incl)
    if algo == 0x08:
        return _bsum(data, start, end_incl)
    return None


# ====== Slot 检测 ======

# 每个 slot descriptor 起始为 0xDEADBEEF (LE)
DEADBEEF_LE = b'\xEF\xBE\xAD\xDE'
SENTINEL_LEN = 0x9C  # 156 字节零哨兵
SENTINEL = b'\x00' * SENTINEL_LEN

# 三种 layout: (sentinel_off, base_off, length_byte_off, validity_u32_off)
LAYOUTS = [
    (0x004, 0x100, 0x113, 0x104),
    (0x024, 0x120, 0x133, 0x124),
    (0x104, 0x200, 0x213, 0x204),
]
LAYOUT_NAMES = ['A', 'B', 'C']


@dataclass
class SubRegion:
    # 16 字节子区域描述符（slot+0x150 + i*16）
    index: int
    start_addr: int            # TC2xx 绝对地址
    end_addr: int              # TC2xx 绝对地址（CRC32 含端，ADD 不含）
    flag: int                  # +0x08 u16
    bswap_byte: int            # +0x0A 字节序翻转标志
    algorithm: int             # +0x0B 算法 ID
    stored_crc_addr: int       # +0x0C 存储位置绝对地址
    start_off: Optional[int] = None
    end_off:   Optional[int] = None
    crc_off:   Optional[int] = None


@dataclass
class ChecksumSlot:
    # 4KB 对齐 slot，含外层 header CRC + 内层 sub-region 列表
    descriptor_off: int
    layout: int
    crc_start: int             # 外层 CRC 起始
    crc_end: int               # 外层 CRC 结束（含）
    stored_crc_off: int
    stored_inv_off: int
    stored_crc: int
    stored_inv: int
    length_factor: int         # 子区域数量
    algo_byte: int             # +0x112 (slot 级算法提示)
    sw_id: str                 # +0x144 起 10 字节 ASCII
    end_of_data_addr: int      # slot+0x04，FA*8 trailer 起始绝对地址
    crosslink_addr: int        # slot+0x14，指向另一 slot 的 trailer+8
    sub_regions: List[SubRegion] = field(default_factory=list)

    @property
    def is_consistent(self) -> bool:
        # 存储 CRC 与其反码必须互补，否则 slot 未编程
        return self.stored_crc == (~self.stored_inv & 0xFFFFFFFF)

    @property
    def header_size(self) -> int:
        return self.crc_end - self.crc_start + 1


def _parse_sub_regions(data: bytes, slot: ChecksumSlot) -> List[SubRegion]:
    subs: List[SubRegion] = []
    for i in range(slot.length_factor):
        off = slot.descriptor_off + 0x150 + i * 16
        if off + 16 > len(data):
            break
        start_addr = struct.unpack_from('<I', data, off)[0]
        end_addr   = struct.unpack_from('<I', data, off + 4)[0]
        flag       = struct.unpack_from('<H', data, off + 8)[0]
        bswap_byte = data[off + 0x0A]
        algorithm  = data[off + 0x0B]
        crc_addr   = struct.unpack_from('<I', data, off + 0x0C)[0]
        subs.append(SubRegion(
            index=i, start_addr=start_addr, end_addr=end_addr, flag=flag,
            bswap_byte=bswap_byte, algorithm=algorithm, stored_crc_addr=crc_addr,
            start_off=addr_to_offset(start_addr),
            end_off=addr_to_offset(end_addr),
            crc_off=addr_to_offset(crc_addr),
        ))
    return subs


def find_slots(data: bytes, step: int = 0x1000) -> Iterator[ChecksumSlot]:
    # 4KB 步长扫描，DEADBEEF 预筛 + 哨兵确认
    n = len(data)
    for desc in range(0, n, step):
        if data[desc:desc+4] != DEADBEEF_LE:
            continue
        for idx, (sent_off, base_off, lenb_off, valid_off) in enumerate(LAYOUTS):
            sent_pos = desc + sent_off
            if sent_pos + SENTINEL_LEN > n or desc + valid_off + 4 > n:
                continue
            if data[sent_pos:sent_pos + SENTINEL_LEN] != SENTINEL:
                continue
            # 有效性字 != 0（区分已编程区域 / 区分 layout A vs C）
            if struct.unpack_from('<I', data, desc + valid_off)[0] == 0:
                continue
            length_units = data[desc + lenb_off]
            crc_start = desc + base_off
            crc_end = desc + base_off + length_units * 0x10 + 0x4F
            stored_crc_off = crc_end + 1
            stored_inv_off = crc_end + 5
            if stored_inv_off + 4 > n:
                continue
            slot = ChecksumSlot(
                descriptor_off=desc, layout=idx,
                crc_start=crc_start, crc_end=crc_end,
                stored_crc_off=stored_crc_off, stored_inv_off=stored_inv_off,
                stored_crc=struct.unpack_from('<I', data, stored_crc_off)[0],
                stored_inv=struct.unpack_from('<I', data, stored_inv_off)[0],
                length_factor=length_units,
                algo_byte=data[desc + base_off + 0x12],
                sw_id=data[desc + base_off + 0x44:desc + base_off + 0x4E]
                       .split(b'\x00')[0].decode('ascii', errors='replace'),
                end_of_data_addr=struct.unpack_from('<I', data, desc + base_off + 0x04)[0],
                crosslink_addr=struct.unpack_from('<I', data, desc + base_off + 0x14)[0],
            )
            slot.sub_regions = _parse_sub_regions(data, slot)
            yield slot
            break


# ====== Trailer 结构校验 ======

# Trailer 布局（位于每个数据区末尾，绝对地址 = slot+0x04 字段）：
#   +0x00..+0x07: FA*8 哨兵
#   +0x08..+0x0F: 8 字节（与 slot+0x18..+0x1F 镜像）
#   +0x10:        0x00000100
#   +0x14:        0x000002A2 或 0x000002A3
#   +0x18..+0x3F: 40 字节签名（DLL 不验，由 ECU HSM 处理）
TRAILER_SENTINEL = b'\xFA' * 8


def validate_trailer(data: bytes, trailer_off: int) -> bool:
    # 仅做结构校验（FUN_10001680），不动 40B 签名
    if trailer_off + 0x18 > len(data):
        return False
    if data[trailer_off:trailer_off + 8] != TRAILER_SENTINEL:
        return False
    if struct.unpack_from('<I', data, trailer_off + 0x10)[0] != 0x00000100:
        return False
    v14 = struct.unpack_from('<I', data, trailer_off + 0x14)[0]
    return v14 in (0x000002A2, 0x000002A3)


# ====== Verify 与 Recompute ======

@dataclass
class CheckResult:
    layer: str             # "L1-header" 或 "L2-subregion"
    slot_off: int
    sub_index: Optional[int]
    algorithm: int
    crc_range: tuple
    stored: int
    computed: Optional[int]
    ok: bool
    note: str = ""


def verify(data: bytes) -> List[CheckResult]:
    results: List[CheckResult] = []
    for slot in find_slots(data):
        # 第 1 层：slot header CRC
        if slot.is_consistent:
            l1 = _crc32(data, slot.crc_start, slot.crc_end)
            results.append(CheckResult(
                layer="L1-header", slot_off=slot.descriptor_off, sub_index=None,
                algorithm=ALGO_CRC32,
                crc_range=(slot.crc_start, slot.crc_end),
                stored=slot.stored_crc, computed=l1,
                ok=(l1 == slot.stored_crc),
                note=f"sw={slot.sw_id} layout={LAYOUT_NAMES[slot.layout]}",
            ))
        else:
            results.append(CheckResult(
                layer="L1-header", slot_off=slot.descriptor_off, sub_index=None,
                algorithm=ALGO_CRC32,
                crc_range=(slot.crc_start, slot.crc_end),
                stored=slot.stored_crc, computed=None, ok=False,
                note="header 不一致 (CRC 与 ~CRC 不互补)，跳过",
            ))
            continue

        # 第 2 层：子区域逐项
        for sub in slot.sub_regions:
            if sub.start_off is None or sub.end_off is None or sub.crc_off is None:
                results.append(CheckResult(
                    layer="L2-subregion", slot_off=slot.descriptor_off,
                    sub_index=sub.index, algorithm=sub.algorithm,
                    crc_range=(sub.start_addr, sub.end_addr),
                    stored=0, computed=None, ok=False,
                    note="地址转换失败",
                ))
                continue
            if sub.algorithm == 0xFF:
                results.append(CheckResult(
                    layer="L2-subregion", slot_off=slot.descriptor_off,
                    sub_index=sub.index, algorithm=sub.algorithm,
                    crc_range=(sub.start_off, sub.end_off),
                    stored=0, computed=None, ok=True,
                    note="WRMG 写魔数算法，无需验证",
                ))
                continue
            stored = struct.unpack_from('<I', data, sub.crc_off)[0]
            if sub.bswap_byte:
                stored = bswap32(stored)
            computed = compute_algo(sub.algorithm, data, sub.start_off, sub.end_off)
            ok = (computed is not None and computed == stored)
            note = "" if computed is not None else f"未知算法 0x{sub.algorithm:02X}"
            results.append(CheckResult(
                layer="L2-subregion", slot_off=slot.descriptor_off,
                sub_index=sub.index, algorithm=sub.algorithm,
                crc_range=(sub.start_off, sub.end_off),
                stored=stored, computed=computed, ok=ok, note=note,
            ))
    return results


def recompute(data: bytes) -> bytes:
    # 先重算第 2 层，再重算第 1 层（顺序无关，因 L1 不覆盖 L2 存储位置）
    buf = bytearray(data)
    slots = list(find_slots(data))

    for slot in slots:
        if not slot.is_consistent:
            continue
        for sub in slot.sub_regions:
            if sub.algorithm == 0xFF:
                continue
            if sub.start_off is None or sub.end_off is None or sub.crc_off is None:
                continue
            new_val = compute_algo(sub.algorithm, buf, sub.start_off, sub.end_off)
            if new_val is None:
                continue
            stored_val = bswap32(new_val) if sub.bswap_byte else new_val
            struct.pack_into('<I', buf, sub.crc_off, stored_val)

    for slot in slots:
        if not slot.is_consistent:
            continue
        new_crc = _crc32(buf, slot.crc_start, slot.crc_end)
        struct.pack_into('<I', buf, slot.stored_crc_off, new_crc)
        struct.pack_into('<I', buf, slot.stored_inv_off, (~new_crc) & 0xFFFFFFFF)

    return bytes(buf)


# ====== DLL 内置支持的 SW 变种表（PTR_PTR_1000ae30）======

SUPPORTED_SW_VARIANTS = [
    "MG1CS003", "MG1CS042",
    "MD1_CS004", "MD1CS003", "MD1CS006",
    "MDG1_MD1CP032",
    "RBA1DGS1/VWAUDI2", "RBA1DGS1/VWAUDI3", "RBA1DGS1/VWAUDI4",
    "RBA1DGS1/VWAUDI5", "RBA1DGS1/VWAUDI6", "RBA1DGS1/VWAUDI7",
    "RBA1DGS1/VWAUDI8", "RBA1DGS1/VWAUDI9",
    "RBA1DGS1/CNHI1", "RBA1DGS1/DAI1", "RBA1DGS1/RSA1",
]


# ====== CLI ======

def _format_result(r: CheckResult) -> str:
    algo = ALGO_NAMES.get(r.algorithm, f"?{r.algorithm:02X}")
    sa, ea = r.crc_range
    rng = f"0x{sa:06X}..0x{ea:06X}"
    sz = ea - sa + 1
    cs = f"0x{r.computed:08X}" if r.computed is not None else "(none)"
    status = "OK" if r.ok else "*** FAIL ***"
    sub = f" sub#{r.sub_index:>2}" if r.sub_index is not None else "       "
    return (f"  [{r.layer:<13}] slot=0x{r.slot_off:06X}{sub}  {algo:<5} "
            f"{rng}  ({sz:>9,} B)  stored=0x{r.stored:08X} computed={cs:<10}  "
            f"{status}  {r.note}")


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Bosch MDG1 TC2xx checksum (build by dukeapenguin)")
    ap.add_argument("bin", help="ECU bin 文件路径（通常 8MB）")
    ap.add_argument("-w", "--write", metavar="OUTPUT",
                    help="重算所有 checksum 并写到 OUTPUT")
    ap.add_argument("--quiet", action="store_true",
                    help="只显示失败项 + 汇总")
    ap.add_argument("--trailers", action="store_true",
                    help="同时校验 trailer 结构")
    args = ap.parse_args()

    with open(args.bin, "rb") as f:
        data = f.read()

    print(f"File: {args.bin}  ({len(data):,} bytes)\n")
    results = verify(data)
    slots = list(find_slots(data))
    n_l1 = sum(1 for r in results if r.layer == "L1-header")
    n_l2 = sum(1 for r in results if r.layer == "L2-subregion")
    print(f"找到 {len(slots)} 个 slot, L1 header 校验 {n_l1} 项, L2 sub-region 校验 {n_l2} 项")

    bad = []
    for r in results:
        if not r.ok:
            bad.append(r)
        if not args.quiet or not r.ok:
            print(_format_result(r))

    print(f"\n汇总: {len(results) - len(bad)} OK / {len(bad)} FAIL")

    if args.trailers:
        print("\n--- Trailer 结构校验 (FUN_10001680) ---")
        for slot in slots:
            t_off = addr_to_offset(slot.end_of_data_addr)
            if t_off is None:
                print(f"  slot 0x{slot.descriptor_off:06X}: end-of-data 地址 "
                      f"0x{slot.end_of_data_addr:08X} 不可映射")
                continue
            ok = validate_trailer(data, t_off)
            xlink = "(none)"
            if slot.crosslink_addr != 0xFAFAFAFA:
                xt = addr_to_offset(slot.crosslink_addr - 8) if slot.crosslink_addr >= 8 else None
                xlink = f"0x{xt:06X}" if xt is not None else f"unmappable 0x{slot.crosslink_addr:08X}"
            print(f"  slot 0x{slot.descriptor_off:06X}: trailer @ 0x{t_off:06X}  "
                  f"{'OK' if ok else 'FAIL'}   xlink->{xlink}")

    if args.write:
        new_data = recompute(data)
        with open(args.write, "wb") as f:
            f.write(new_data)
        new_bad = [r for r in verify(new_data) if not r.ok]
        print(f"\n写入 {args.write}  (重算后剩余 {len(new_bad)} 个 FAIL)")

    return 1 if (bad and not args.write) else 0


if __name__ == "__main__":
    raise SystemExit(_main())
