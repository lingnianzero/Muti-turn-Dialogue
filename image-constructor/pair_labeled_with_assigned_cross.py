from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple


LABELED_ROOT_DEFAULT = Path("human_portrait_labeled")
ASSIGNED_ROOT_DEFAULT = Path("human_portrait_assigned")


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise SystemExit(f"JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def full_cross_pair(labeled_root: Path, assigned_root: Path) -> Tuple[int, int]:
    """Assign for every labeled entry a cross-category assigned image, distributing target categories evenly
    and avoiding symmetric pairs (A->B and B->A).
    Returns (paired_count, total_entries).
    """
    labeled_map_path = labeled_root / "labeled_map.json"
    assigned_map_path = assigned_root / "assigned_map.json"

    labeled = load_json(labeled_map_path)
    assigned = load_json(assigned_map_path)

    l_entries: List[Dict] = labeled.get("entries", [])
    a_entries: List[Dict] = assigned.get("entries", [])

    # Index: assigned path -> labeled index (if exists)
    assigned_to_lidx: Dict[str, int] = {}
    for i, e in enumerate(l_entries):
        ap = e.get("assigned")
        if ap:
            assigned_to_lidx[ap] = i

    # Group labeled by category
    cats = sorted({e.get("category") for e in l_entries if e.get("category")})
    l_by_cat: Dict[str, List[int]] = defaultdict(list)
    for i, e in enumerate(l_entries):
        c = e.get("category")
        if c:
            l_by_cat[c].append(i)

    # Candidates grouped by target category (as queues)
    cand_by_cat: Dict[str, Deque[Dict]] = defaultdict(deque)
    for a in a_entries:
        c = a.get("category")
        if c:
            cand_by_cat[c].append(a)

    # Track: for avoiding symmetric pairs and overuse
    target_used: set[str] = set()  # assigned path used as target
    paired_source_for_target: Dict[str, str] = {}  # target_assigned -> source_assigned

    # For even distribution: per source category, count how many times each target category chosen
    dist_count: Dict[str, Dict[str, int]] = {c: {t: 0 for t in cats if t != c} for c in cats}

    def choose_target_category(src_cat: str) -> List[str]:
        # Return other categories sorted by least used first (for balancing)
        other = [t for t in cats if t != src_cat]
        other.sort(key=lambda t: dist_count[src_cat][t])
        return other

    def valid_candidate(src_ent: Dict, cand: Dict) -> bool:
        if not cand:
            return False
        if cand.get("category") == src_ent.get("category"):
            return False
        if cand.get("timestamp") == src_ent.get("timestamp"):
            return False
        if cand.get("name") == src_ent.get("name"):
            return False
        ap_src = src_ent.get("assigned")
        ap_cand = cand.get("assigned")
        if not (ap_src and ap_cand):
            return False
        # Avoid symmetric: do not pick candidate equal to the source that previously used us as target
        forbid = paired_source_for_target.get(ap_src)
        if forbid and ap_cand == forbid:
            return False
        # Avoid reusing the same target assigned multiple times
        if ap_cand in target_used:
            return False
        return True

    paired = 0
    total = len(l_entries)

    for src_cat in cats:
        idxs = l_by_cat.get(src_cat, [])
        for i in idxs:
            ent = l_entries[i]
            # Try categories by balance order
            picked = None
            order = choose_target_category(src_cat)
            for tgt_cat in order:
                # Try candidates within target category
                tried = 0
                limit = len(cand_by_cat[tgt_cat])
                while tried < limit and cand_by_cat[tgt_cat]:
                    cand = cand_by_cat[tgt_cat][0]
                    # Rotate to avoid starving
                    cand_by_cat[tgt_cat].rotate(-1)
                    tried += 1
                    if valid_candidate(ent, cand):
                        picked = cand
                        break
                if picked:
                    dist_count[src_cat][tgt_cat] += 1
                    break
            # Fallback across all categories if not found
            if not picked:
                for tgt_cat in (t for t in cats if t != src_cat):
                    for cand in list(cand_by_cat[tgt_cat])[: min(50, len(cand_by_cat[tgt_cat]))]:
                        if valid_candidate(ent, cand):
                            picked = cand
                            dist_count[src_cat][tgt_cat] += 1
                            break
                    if picked:
                        break

            if picked:
                ap_src = ent.get("assigned")
                ap_tgt = picked.get("assigned")
                ent["cross_assigned"] = {
                    "assigned": ap_tgt,
                    "category": picked.get("category"),
                    "id": picked.get("id"),
                    "timestamp": picked.get("timestamp"),
                    "name": picked.get("name"),
                }
                target_used.add(ap_tgt)
                paired_source_for_target[ap_tgt] = ap_src  # For symmetric avoidance
                paired += 1
            else:
                # No candidate met constraints; clear any previous to be explicit
                ent["cross_assigned"] = None

    # Second pass: fill any remaining, still balancing across categories and preferring unused targets.
    # Maintain per-target-category rotation pointers by reusing cand_by_cat deques and rotating on each pick.
    missing_idx = [i for i, e in enumerate(l_entries) if not e.get("cross_assigned")]
    if missing_idx:
        for i in missing_idx:
            ent = l_entries[i]
            src_cat = ent.get("category")
            ap_src = ent.get("assigned")
            forbid = paired_source_for_target.get(ap_src)
            picked = None

            # Try balanced order first, prefer candidates not yet used
            for tgt_cat in (t for t in choose_target_category(src_cat) if t != src_cat):
                tried = 0
                limit = len(cand_by_cat[tgt_cat])
                # One sweep for unused targets
                while tried < limit and cand_by_cat[tgt_cat]:
                    cand = cand_by_cat[tgt_cat][0]
                    cand_by_cat[tgt_cat].rotate(-1)
                    tried += 1
                    ap_tgt = cand.get("assigned")
                    if not ap_tgt:
                        continue
                    if ap_tgt in target_used:
                        continue
                    if (
                        cand.get("category") != src_cat
                        and cand.get("timestamp") != ent.get("timestamp")
                        and cand.get("name") != ent.get("name")
                        and not (forbid and ap_tgt == forbid)
                    ):
                        picked = cand
                        break
                if picked:
                    break

            # If still not found, allow reuse but keep constraints and rotation to avoid bias
            if not picked:
                for tgt_cat in (t for t in cats if t != src_cat):
                    tried = 0
                    limit = len(cand_by_cat[tgt_cat])
                    while tried < limit and cand_by_cat[tgt_cat]:
                        cand = cand_by_cat[tgt_cat][0]
                        cand_by_cat[tgt_cat].rotate(-1)
                        tried += 1
                        ap_tgt = cand.get("assigned")
                        if not ap_tgt:
                            continue
                        if (
                            cand.get("category") != src_cat
                            and cand.get("timestamp") != ent.get("timestamp")
                            and cand.get("name") != ent.get("name")
                            and not (forbid and ap_tgt == forbid)
                        ):
                            picked = cand
                            break
                    if picked:
                        break

            if picked:
                ap_tgt = picked.get("assigned")
                ent["cross_assigned"] = {
                    "assigned": ap_tgt,
                    "category": picked.get("category"),
                    "id": picked.get("id"),
                    "timestamp": picked.get("timestamp"),
                    "name": picked.get("name"),
                }
                # Mark as used if it wasn't yet to improve fairness for later picks
                if ap_tgt not in target_used:
                    target_used.add(ap_tgt)
                paired += 1

    save_json(labeled_map_path, {"total": total, "entries": l_entries})
    return paired, total


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create cross-category pairing for all labeled images with assigned images.")
    ap.add_argument("--labeled-root", type=Path, default=LABELED_ROOT_DEFAULT)
    ap.add_argument("--assigned-root", type=Path, default=ASSIGNED_ROOT_DEFAULT)
    ap.add_argument("--mode", choices=["full", "single"], default="full", help="full: pair all; single: pair one example")
    return ap.parse_args()


def main() -> None:
    ns = parse_args()
    if ns.mode == "full":
        paired, total = full_cross_pair(ns.labeled_root, ns.assigned_root)
        print(f"Cross paired {paired}/{total} labeled entries (balanced by target categories, symmetric avoided).")
    else:
        # Legacy single example path kept for debugging
        labeled_map_path = ns.labeled_root / "labeled_map.json"
        assigned_map_path = ns.assigned_root / "assigned_map.json"
        from random import shuffle
        labeled_data = load_json(labeled_map_path)
        assigned_data = load_json(assigned_map_path)
        l_entries = labeled_data.get("entries", [])
        a_entries = assigned_data.get("entries", [])
        # simple pick
        for i, ent in enumerate(l_entries):
            for cand in a_entries:
                if (
                    cand.get("category") != ent.get("category")
                    and cand.get("timestamp") != ent.get("timestamp")
                    and cand.get("name") != ent.get("name")
                ):
                    ent["cross_assigned"] = {
                        "assigned": cand.get("assigned"),
                        "category": cand.get("category"),
                        "id": cand.get("id"),
                        "timestamp": cand.get("timestamp"),
                        "name": cand.get("name"),
                    }
                    save_json(labeled_map_path, {"total": len(l_entries), "entries": l_entries})
                    print("Paired one example.")
                    return
        print("No example paired.")


if __name__ == "__main__":
    main()
