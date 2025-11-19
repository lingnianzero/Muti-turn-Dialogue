from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image


SRC_DEFAULT = Path("baidu_human_portrait_filtered")
DST_DEFAULT = Path("human_portrait_raw")


# 中文目录到英文目录名的映射
ZH2EN: Dict[str, str] = {
    "校园写真": "campus_portrait",
    "人物特写": "closeup_portrait",
    "人物头像": "headshot_portrait",
    "室内人像": "indoor_portrait",
    "街头人像": "street_portrait",
    "儿童肖像": "child_portrait",
    "生活方式摄影": "lifestyle_portrait",
    "老人肖像": "senior_portrait",
    "旅行写真": "travel_portrait",
    "职场肖像": "workplace_portrait",
}


# 解析 jpg 文件名中的毫秒时间戳：中文_1762637215128_7122.jpg
TS_RE = re.compile(r"^.+?_(?P<ts>\d{13})_\d{4}\.jpe?g$", re.IGNORECASE)


@dataclass(frozen=True)
class Item:
    eng: str
    ts: str
    path: Path  # 源 JPG 路径


def iter_source(src_root: Path) -> List[Item]:
    items: List[Item] = []
    for zh_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        eng = ZH2EN.get(zh_dir.name)
        if not eng:
            # 非预期目录，跳过
            continue
        # 兼容 jpg/jpeg 两种扩展
        for p in sorted(list(zh_dir.glob("*.jpg")) + list(zh_dir.glob("*.jpeg"))):
            m = TS_RE.match(p.name)
            if not m:
                # 非预期文件名，跳过
                continue
            ts = m.group("ts")
            items.append(Item(eng=eng, ts=ts, path=p))
    # 全局稳定排序：英文类目、时间戳、文件名
    items.sort(key=lambda it: (it.eng, int(it.ts), it.path.name))
    return items


def ensure_dst(dst_root: Path, categories: Iterable[str]) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    for cat in sorted(set(categories)):
        (dst_root / cat).mkdir(parents=True, exist_ok=True)


def clean_old(dst_root: Path) -> Tuple[int, int]:
    """删除 human_portrait_raw 下旧图片（png/jpg/jpeg），返回删除计数。"""
    removed = 0
    files = 0
    if not dst_root.exists():
        return removed, files
    for p in dst_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            files += 1
            try:
                p.unlink()
                removed += 1
            except Exception:
                pass
    return removed, files


def convert_all(
    items: List[Item],
    dst_root: Path,
    start_id: int = 0,
) -> Tuple[int, List[Dict[str, str]]]:
    mapping: List[Dict[str, str]] = []
    ensure_dst(dst_root, (it.eng for it in items))

    cur_id = start_id
    for it in items:
        new_name = f"{it.eng}_RAW_{cur_id}_{it.ts}.png"
        dst_path = dst_root / it.eng / new_name
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(it.path)
        img.save(dst_path, format="PNG")

        mapping.append(
            {
                "src": f"{it.path.parent.name}/{it.path.name}",  # 相对 src 根
                "dst": f"{it.eng}/{new_name}",  # 相对 dst 根
                "category": it.eng,
                "timestamp": it.ts,
                "id": cur_id,
            }
        )
        cur_id += 1
    return cur_id - start_id, mapping


def write_mapping(dst_root: Path, mapping: List[Dict[str, str]]) -> Path:
    out = dst_root / "rename_map.json"
    out.write_text(json.dumps({"total": len(mapping), "entries": mapping}, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert filtered JPGs to RAW PNG naming in human_portrait_raw.")
    ap.add_argument("--src", type=Path, default=SRC_DEFAULT, help="Source root (default: baidu_human_portrait_filtered)")
    ap.add_argument("--dst", type=Path, default=DST_DEFAULT, help="Destination root (default: human_portrait_raw)")
    ap.add_argument("--start-id", type=int, default=0, help="Global starting ID (default: 0)")
    ap.add_argument("--no-clean", action="store_true", help="Do not delete old images under destination root")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N images (for testing)")
    return ap.parse_args()


def main() -> None:
    ns = parse_args()
    if not ns.src.exists():
        raise SystemExit(f"Source not found: {ns.src}")
    items = iter_source(ns.src)
    if not items:
        raise SystemExit("No JPG images found.")
    if ns.limit is not None and ns.limit > 0:
        items = items[: ns.limit]

    if not ns.no_clean:
        removed, total = clean_old(ns.dst)
        print(f"Cleaned old images in {ns.dst}: removed={removed}/{total}")

    count, mapping = convert_all(items, ns.dst, ns.start_id)
    out_map = write_mapping(ns.dst, mapping)
    print(f"Converted {count} images. Mapping saved to: {out_map}")


if __name__ == "__main__":
    main()
