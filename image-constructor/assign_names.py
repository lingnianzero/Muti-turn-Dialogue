from __future__ import annotations

import argparse
import json
import re
import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# 统一姓名池源于当前项目的 name_generator
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from name_generator import NAMES as NAME_POOL


SRC_DEFAULT = Path("human_portrait_raw")
DST_DEFAULT = Path("human_portrait_assigned")

# 原 RAW 文件名：<eng>_RAW_<ID>_<ts>.png
RAW_REGEX = re.compile(r"^(?P<eng>[a-z_]+)_RAW_(?P<id>\d+)_(?P<ts>\d{13})\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class RawItem:
    eng: str
    id: int
    ts: str
    path: Path  # 源 RAW 文件全路径


def iter_raw_images(src_root: Path) -> Dict[str, List[RawItem]]:
    """按英文类目分组扫描 RAW 图片。返回 {eng: [RawItem, ...]} 排序好的列表。"""
    groups: Dict[str, List[RawItem]] = {}

    for sub in sorted(src_root.iterdir()):
        if not sub.is_dir() or not sub.name.endswith("_portrait"):
            continue
        items: List[RawItem] = []
        for p in sorted(sub.glob("*.png")):
            m = RAW_REGEX.match(p.name)
            if not m:
                # 非规范文件名，跳过
                continue
            eng = m.group("eng")
            rid = int(m.group("id"))
            ts = m.group("ts")
            items.append(RawItem(eng=eng, id=rid, ts=ts, path=p))
        # 按 ID、时间戳、文件名稳定排序
        items.sort(key=lambda it: (it.id, int(it.ts), it.path.name))
        if items:
            groups[sub.name] = items  # 使用目录名（如 campus_portrait）作为键
    return groups


def ensure_dst_structure(dst_root: Path, categories: Iterable[str]) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    for cat in sorted(set(categories)):
        (dst_root / cat).mkdir(parents=True, exist_ok=True)


def make_name_sequence(count: int, shuffle: bool, seed: Optional[int], eng: str) -> List[str]:
    """根据规则生成姓名序列：
    - 若 count <= 500（姓名池大小），保证不重复
    - 若 count > 500，先用满 500 个，再从头循环
    - 默认可洗牌；若提供 seed，则对每个子目录使用独立但可复现的洗牌
    """
    base = list(NAME_POOL)
    # NAME_POOL 已按字母序；根据需要洗牌
    if shuffle:
        s = (seed if seed is not None else 0) ^ zlib.crc32(eng.encode("utf-8"))
        import random

        rnd = random.Random(s)
        rnd.shuffle(base)

    if count <= len(base):
        return base[:count]
    # 超出则循环分配
    seq: List[str] = []
    full, rem = divmod(count, len(base))
    for _ in range(full):
        seq.extend(base)
    if rem:
        seq.extend(base[:rem])
    return seq


def assign_and_copy(
    groups: Dict[str, List[RawItem]],
    dst_root: Path,
    shuffle: bool,
    seed: Optional[int],
    limit: Optional[int] = None,
) -> Tuple[int, List[Dict[str, str]]]:
    """执行分配与复制，返回处理总数与映射条目。"""
    mapping: List[Dict[str, str]] = []
    processed = 0

    # 加载 RAW 阶段的映射，便于合并
    raw_map_path = SRC_DEFAULT / "rename_map.json"
    raw_map_index: Dict[str, Dict[str, str]] = {}
    if raw_map_path.exists():
        raw_data = json.loads(raw_map_path.read_text(encoding="utf-8"))
        for ent in raw_data.get("entries", []):
            # 键：相对 RAW 根目录的路径
            raw_map_index[str(ent.get("dst", ""))] = ent

    for cat, items in groups.items():
        names = make_name_sequence(len(items), shuffle=shuffle, seed=seed, eng=cat)
        for i, item in enumerate(items):
            if limit is not None and processed >= limit:
                return processed, mapping

            name = names[i]
            # 新文件名：<eng>_<name>_<ID>_<ts>.png
            new_name = f"{item.eng}_{name}_{item.id}_{item.ts}.png"
            rel_src = item.path.relative_to(SRC_DEFAULT)
            dst_path = dst_root / cat / new_name
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item.path, dst_path)

            rel_dst = dst_path.relative_to(dst_root)

            # 合并 RAW 映射
            raw_ent = raw_map_index.get(str(rel_src).replace("\\", "/"), {})
            mapping.append(
                {
                    "orig": raw_ent.get("src"),  # 原始（生成前）路径，若可得
                    "raw": str(rel_src).replace("\\", "/"),
                    "assigned": str(rel_dst).replace("\\", "/"),
                    "category": item.eng,
                    "id": item.id,
                    "timestamp": item.ts,
                    "name": name,
                }
            )
            processed += 1

    return processed, mapping


def write_mapping(dst_root: Path, mapping: List[Dict[str, str]]) -> Path:
    out = dst_root / "assigned_map.json"
    data = {"total": len(mapping), "entries": mapping}
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Assign names to RAW portraits and copy to human_portrait_assigned."
    )
    ap.add_argument("--src", type=Path, default=SRC_DEFAULT, help="Source RAW root (default: human_portrait_raw)")
    ap.add_argument(
        "--dst", type=Path, default=DST_DEFAULT, help="Destination root (default: human_portrait_assigned)"
    )
    ap.add_argument("--limit", type=int, default=None, help="Process at most N images for quick test")
    ap.add_argument("--seed", type=int, default=None, help="Shuffle seed for per-category name order")
    ap.add_argument(
        "--no-shuffle", dest="shuffle", action="store_false", help="Disable shuffling; keep alphabetical name order"
    )
    ap.set_defaults(shuffle=True)
    return ap.parse_args()


def main() -> None:
    ns = parse_args()
    if not ns.src.exists():
        raise SystemExit(f"Source folder not found: {ns.src}")

    groups = iter_raw_images(ns.src)
    if not groups:
        raise SystemExit(f"No RAW images found under {ns.src}")

    ensure_dst_structure(ns.dst, groups.keys())
    total, mapping = assign_and_copy(groups, ns.dst, shuffle=ns.shuffle, seed=ns.seed, limit=ns.limit)
    out_path = write_mapping(ns.dst, mapping)
    print(f"Assigned {total} images. Mapping saved to: {out_path}")


if __name__ == "__main__":
    main()
