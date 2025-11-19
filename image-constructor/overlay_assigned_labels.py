from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from PIL import Image, ImageDraw, ImageFont


# 默认路径
ASSIGNED_ROOT_DEFAULT = Path("human_portrait_assigned")
LABELED_ROOT_DEFAULT = Path("human_portrait_labeled")
CONFIG_PATH_DEFAULT = Path("config.yaml")


@dataclass
class LabelStyle:
    """图片文字标签样式（与 ImageProcessor 中保持一致字段）。"""

    font_path: str
    font_size: int
    padding: int
    text_color: Sequence[int]
    background_color: Sequence[int]

    @classmethod
    def from_config(cls, cfg: Dict) -> "LabelStyle":
        return cls(
            font_path=cfg.get("font_path", "/System/Library/Fonts/Supplemental/Arial.ttf"),
            font_size=int(cfg.get("font_size", 120)),
            padding=int(cfg.get("padding", 40)),
            text_color=tuple(cfg.get("text_color", [0, 0, 0])),
            background_color=tuple(cfg.get("background_color", [255, 255, 255])),
        )

    def load_font(self) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        try:
            return ImageFont.truetype(self.font_path, self.font_size)
        except OSError:
            logging.warning("字体 %s 无法加载，退回默认字体。", self.font_path)
            return ImageFont.load_default()


def load_label_style(config_path: Path) -> LabelStyle:
    """从 config.yaml 读取 processing.label_style，若不可用则使用默认。"""
    cfg: Dict = {}
    if config_path.exists():
        try:
            import yaml  # type: ignore

            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            logging.warning("读取配置失败，将使用默认样式: %s", exc)
    style_cfg = (cfg.get("processing", {}) or {}).get("label_style", {})
    return LabelStyle.from_config(style_cfg)


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    """兼容不同 Pillow 版本的文字尺寸计算（对齐 ImageProcessor 实现）。"""
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    if hasattr(draw, "textlength"):
        width = int(draw.textlength(text, font=font))
        ascent, descent = font.getmetrics()
        height = ascent + descent
        return width, height
    return draw.textsize(text, font=font)


def _load_font_with_size(path: str, size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        logging.warning("字体 %s 无法按大小 %s 加载，退回默认字体。", path, size)
        return ImageFont.load_default()


def _fit_font_for_text(
    image_width: int, text: str, style: LabelStyle, base_font: ImageFont.ImageFont
) -> ImageFont.ImageFont:
    """根据图片宽度自适应缩小字号，确保文字完全显示。

    策略：
    - 目标宽度 = image_width - 2*padding
    - 若文字宽度超出，按比例缩放指定字号，最小不低于 12px
    - 最多进行少量迭代以确保完全贴合
    """
    target_width = max(10, image_width - style.padding * 2)

    # 若 base_font 可提供字号信息，尝试按比例缩放
    size_guess = getattr(style, "font_size", 160)
    draw_tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    w, h = measure_text(draw_tmp, text, base_font)
    if w <= target_width:
        return base_font

    # 比例缩放估算
    ratio = target_width / float(w) if w else 1.0
    new_size = max(12, int(size_guess * ratio))
    font = _load_font_with_size(style.font_path, new_size)

    # 细化迭代，避免轻微溢出
    for _ in range(5):
        w2, _ = measure_text(draw_tmp, text, font)
        if w2 <= target_width:
            break
        new_size = max(12, int(new_size * 0.9))
        font = _load_font_with_size(style.font_path, new_size)
    return font


def draw_label(image_path: Path, text: str, style: LabelStyle, font: ImageFont.ImageFont) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    # 尝试按宽度自适应字体大小
    font_fitted = _fit_font_for_text(image.width, text, style, font)
    draw = ImageDraw.Draw(image)
    text_width, text_height = measure_text(draw, text, font_fitted)
    label_height = text_height + style.padding * 2

    canvas = Image.new(
        "RGB",
        (image.width, image.height + label_height),
        color=tuple(style.background_color),
    )
    canvas.paste(image, (0, 0))
    label_draw = ImageDraw.Draw(canvas)
    text_x = (image.width - text_width) // 2
    text_y = image.height + style.padding
    label_draw.text((text_x, text_y), text, fill=tuple(style.text_color), font=font_fitted)
    return canvas


def ensure_dirs(dst_root: Path, categories: Iterable[str]) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    for cat in sorted(set(categories)):
        (dst_root / cat).mkdir(parents=True, exist_ok=True)


def load_assigned_map(assigned_root: Path) -> List[Dict]:
    m = assigned_root / "assigned_map.json"
    if not m.exists():
        raise SystemExit(f"未找到映射文件: {m}")
    data = json.loads(m.read_text(encoding="utf-8"))
    return data.get("entries", [])


def parse_category(assigned_rel_path: str) -> str:
    # 形如 "campus_portrait/campus_portrait_Bethany_1_...png"
    return assigned_rel_path.split("/", 1)[0]


def overlay_all(
    assigned_root: Path,
    labeled_root: Path,
    style: LabelStyle,
    limit: int | None,
    overwrite: bool,
) -> tuple[int, list[dict]]:
    entries = load_assigned_map(assigned_root)
    if not entries:
        return 0, []

    cats = [parse_category(e["assigned"]) for e in entries if e.get("assigned")]
    ensure_dirs(labeled_root, cats)

    font = style.load_font()
    done = 0
    merged: list[dict] = []
    for ent in entries:
        if limit is not None and done >= limit:
            break
        assigned_rel = ent.get("assigned")
        name = ent.get("name")
        if not assigned_rel or not name:
            continue

        src_path = assigned_root / assigned_rel

        # 新命名：<英文条目>_<名字>_labeled_<ID>_<时间戳>.png
        eng = ent.get("category") or parse_category(assigned_rel)
        img_id = ent.get("id")
        ts = ent.get("timestamp")
        dst_filename = f"{eng}_{name}_labeled_{img_id}_{ts}.png"
        dst_path = labeled_root / parse_category(assigned_rel) / dst_filename
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        labeled_rel = f"{parse_category(assigned_rel)}/{dst_filename}"

        if not (dst_path.exists() and not overwrite):
            img = draw_label(src_path, name, style, font)
            # 保持 PNG 扩展名更安全（无损），若后缀是 .jpg 也允许保存 JPEG
            suffix = dst_path.suffix.lower()
            if suffix in {".jpg", ".jpeg"}:
                img.save(dst_path, format="JPEG", quality=95)
            else:
                img.save(dst_path, format="PNG")
        done += 1

        # 合并旧映射并加入新文件相对路径
        ent_merged = dict(ent)
        ent_merged["labeled"] = labeled_rel
        merged.append(ent_merged)

    return done, merged


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Overlay assigned names onto images and export labeled copies.")
    ap.add_argument("--assigned-root", type=Path, default=ASSIGNED_ROOT_DEFAULT, help="Path to human_portrait_assigned")
    ap.add_argument("--labeled-root", type=Path, default=LABELED_ROOT_DEFAULT, help="Output root for labeled images")
    ap.add_argument("--config", type=Path, default=CONFIG_PATH_DEFAULT, help="Path to config.yaml for label style")
    ap.add_argument("--font-path", type=str, default=None, help="Override font path (TTF/OTF)")
    ap.add_argument("--font-size", type=int, default=None, help="Override font size in px")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N images")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite if labeled file exists")
    ap.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return ap.parse_args()


def main() -> None:
    ns = parse_args()
    logging.basicConfig(level=getattr(logging, ns.log_level.upper(), logging.INFO))

    style = load_label_style(ns.config)
    # Optional overrides when config.yaml is not writable or using custom fonts
    if ns.font_path:
        style.font_path = ns.font_path
    if ns.font_size:
        style.font_size = ns.font_size
    ns.labeled_root.mkdir(parents=True, exist_ok=True)
    processed, merged = overlay_all(ns.assigned_root, ns.labeled_root, style, ns.limit, ns.overwrite)
    # 写出合并后的映射
    out_map = ns.labeled_root / "labeled_map.json"
    out_map.write_text(
        json.dumps({"total": processed, "entries": merged}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Labeled {processed} images into: {ns.labeled_root}. Mapping: {out_map}")


if __name__ == "__main__":
    main()
