from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


GEN_DEFAULT = Path("baidu_human_portrait_generated_qwen")
LABELED_DEFAULT = Path("human_portrait_labeled")

# 生成图命名示例：中文_1762637215128_7122__campus_portrait__p165__seed1015575661.png
GEN_RE = re.compile(r"^.+?_(?P<ts>\d{13})_\d{4}__(?P<eng>[a-z_]+)__p\d{3}__seed\d+\.png$",
                    re.IGNORECASE)


def build_generated_index(gen_root: Path) -> Dict[Tuple[str, str], str]:
    """构建 (eng, timestamp) -> relative_path 映射。"""
    index: Dict[Tuple[str, str], str] = {}
    for sub in sorted(gen_root.iterdir()):
        if not sub.is_dir() or not sub.name.endswith("_portrait"):
            continue
        for p in sub.glob("*.png"):
            m = GEN_RE.match(p.name)
            if not m:
                continue
            ts = m.group("ts")
            eng = m.group("eng")
            key = (eng, ts)
            # 路径相对 gen_root，便于记录
            index[key] = f"{sub.name}/{p.name}"
    return index


def load_labeled_map(labeled_root: Path) -> Dict:
    m = labeled_root / "labeled_map.json"
    if not m.exists():
        raise SystemExit(f"未找到映射文件: {m}")
    return json.loads(m.read_text(encoding="utf-8"))


def save_labeled_map(labeled_root: Path, data: Dict) -> Path:
    m = labeled_root / "labeled_map.json"
    m.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return m


def pair_and_update(gen_root: Path, labeled_root: Path) -> Tuple[int, int]:
    index = build_generated_index(gen_root)
    data = load_labeled_map(labeled_root)
    entries: List[Dict] = data.get("entries", [])

    paired = 0
    missing = 0
    for ent in entries:
        eng = ent.get("category")
        ts = ent.get("timestamp")
        if not eng or not ts:
            ent["generated"] = None
            missing += 1
            continue
        key = (eng, ts)
        gen_rel = index.get(key)
        if gen_rel:
            ent["generated"] = gen_rel
            paired += 1
        else:
            ent["generated"] = None
            missing += 1

    save_labeled_map(labeled_root, {"total": len(entries), "entries": entries})
    return paired, missing


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pair labeled images with generated originals by category+timestamp.")
    ap.add_argument("--generated", type=Path, default=GEN_DEFAULT, help="Path to baidu_human_portrait_generated_qwen")
    ap.add_argument("--labeled", type=Path, default=LABELED_DEFAULT, help="Path to human_portrait_labeled")
    return ap.parse_args()


def main() -> None:
    ns = parse_args()
    paired, missing = pair_and_update(ns.generated, ns.labeled)
    print(f"Paired {paired} entries; missing {missing}. Updated: {ns.labeled/'labeled_map.json'}")


if __name__ == "__main__":
    main()

