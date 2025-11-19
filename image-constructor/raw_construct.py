from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import Dict, Iterable, List, Tuple


SRC_DEFAULT = Path("baidu_human_portrait_generated_qwen")
DST_DEFAULT = Path("human_portrait_raw")


# 解析文件名中的毫秒时间戳与英文类目
# 形如：中文类目_1762637215128_7122__campus_portrait__p165__seed1015575661.png
TS_ENG_REGEX = re.compile(
    r"^.+?_(?P<ts>\d{13})_\d{4}__(?P<eng>[a-z_]+)__p\d{3}__seed\d+\.png$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Entry:
    eng: str
    ts: str
    path: Path  # 源文件路径


def iter_source_images(src_root: Path) -> List[Entry]:
    entries: List[Entry] = []
    if not src_root.exists():
        raise SystemExit(f"Source root not found: {src_root}")

    for sub in sorted(src_root.iterdir()):
        if not sub.is_dir():
            continue
        # 仅处理 *_portrait 的十个子目录
        if not sub.name.endswith("_portrait"):
            continue

        for p in sorted(sub.glob("*.png")):
            m = TS_ENG_REGEX.match(p.name)
            if m:
                ts = m.group("ts")
                eng = m.group("eng")
            else:
                # 容错：基于子目录与第一段推断
                parts = p.name.split("__")
                if len(parts) >= 2 and "_" in parts[0]:
                    first = parts[0].split("_")
                    ts = first[1] if len(first) > 1 else "0000000000000"
                else:
                    ts = "0000000000000"
                eng = sub.name

            entries.append(Entry(eng=eng, ts=ts, path=p))

    # 跨目录稳定排序：按英文类目、时间戳、文件名
    entries.sort(key=lambda e: (e.eng, int(e.ts) if e.ts.isdigit() else -1, e.path.name))
    return entries


def ensure_subfolders(dst_root: Path, eng_folders: Iterable[str]) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    for eng in sorted(set(eng_folders)):
        (dst_root / eng).mkdir(parents=True, exist_ok=True)


def build_new_name(eng: str, idx: int, ts: str) -> str:
    return f"{eng}_RAW_{idx}_{ts}.png"


def write_mapping_json(dst_root: Path, mapping: List[Dict[str, str]]) -> Path:
    out = dst_root / "rename_map.json"
    data = {
        "total": len(mapping),
        "entries": mapping,
    }
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def run(
    src_root: Path,
    dst_root: Path,
    start_id: int = 1,
    dry_run: bool = False,
    limit: int | None = None,
) -> Tuple[int, Path]:
    entries = iter_source_images(src_root)
    if not entries:
        raise SystemExit(f"No images found under {src_root}")
    if limit is not None and limit > 0:
        entries = entries[:limit]

    ensure_subfolders(dst_root, (e.eng for e in entries))

    mapping: List[Dict[str, str]] = []
    idx = start_id
    for e in entries:
        new_name = build_new_name(e.eng, idx, e.ts)
        dst_path = dst_root / e.eng / new_name

        rel_src = e.path.relative_to(src_root)
        rel_dst = dst_path.relative_to(dst_root)

        if not dry_run:
            if dst_path.exists():
                raise FileExistsError(f"Destination exists: {dst_path}")
            copyfile(e.path, dst_path)

        mapping.append({
            "src": str(rel_src).replace("\\", "/"),
            "dst": str(rel_dst).replace("\\", "/"),
            "timestamp": e.ts,
            "category": e.eng,
            "id": idx,
        })
        idx += 1

    out_json = write_mapping_json(dst_root, mapping)
    return len(mapping), out_json


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Batch-rename generated portrait images into RAW format.")
    ap.add_argument("--src", type=Path, default=SRC_DEFAULT, help="Source root (default: baidu_human_portrait_generated_qwen)")
    ap.add_argument("--dst", type=Path, default=DST_DEFAULT, help="Destination root (default: human_portrait_raw)")
    ap.add_argument("--start-id", type=int, default=1, help="Starting ID (default: 1)")
    ap.add_argument("--dry-run", action="store_true", help="Do not copy files; only print plan and write nothing")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N images across all categories")
    return ap.parse_args()


def main() -> None:
    ns = parse_args()
    total, out_json = run(ns.src, ns.dst, ns.start_id, ns.dry_run, ns.limit)
    print(f"Renamed {total} images. Mapping saved to: {out_json}")


if __name__ == "__main__":
    main()
