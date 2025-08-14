# translate_images.py
import os
import re
import argparse
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract

from transformers import pipeline
import torch

# ---------- 辅助：中文判断 ----------
CJK_RE = re.compile(r'[\u4e00-\u9fff]')

def has_cjk(s: str) -> bool:
    return bool(CJK_RE.search(s))

# ---------- 辅助：在给定宽度内做“逐字”换行 ----------
def wrap_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int):
    lines = []
    cur = ""
    for ch in text:
        test = cur + ch
        w = draw.textlength(test, font=font)
        if w <= max_width or cur == "":
            cur = test
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines

# ---------- 辅助：根据框里较暗像素估计文字颜色 ----------
def estimate_text_color(bgr_patch: np.ndarray) -> tuple:
    if bgr_patch.size == 0:
        return (0, 0, 0)  # fallback black
    gray = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2GRAY)
    # 取较暗 30% 像素
    thr = np.percentile(gray, 30)
    mask = gray <= thr
    if np.count_nonzero(mask) < 10:
        # 像素太少，直接用整体中位色
        color = np.median(bgr_patch.reshape(-1, 3), axis=0)
    else:
        color = np.median(bgr_patch[mask], axis=0)
    # BGR->RGB
    r, g, b = int(color[2]), int(color[1]), int(color[0])
    return (r, g, b)

# ---------- 主处理：一张图 ----------
def process_one(img_path, out_path, tesseract_cmd=None, font_path=None, translator=None):
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    # 读图
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"[跳过] 无法读取图片：{img_path}")
        return
    h, w = img_bgr.shape[:2]

    # OCR（按行返回框信息）
    # psm 6: Assume a single uniform block of text.
    data = pytesseract.image_to_data(
        img_bgr, lang="chi_sim", output_type=pytesseract.Output.DICT, config="--psm 6"
    )

    n = len(data["text"])
    lines = defaultdict(list)  # key: (block, par, line) -> list of word dict

    for i in range(n):
        txt = data["text"][i].strip()
        conf = int(data["conf"][i]) if data["conf"][i].isdigit() else -1
        if conf < 60:
            continue
        if not txt or not has_cjk(txt):
            # 只处理含中文的词
            continue

        x, y, ww, hh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines[key].append({"text": txt, "x": x, "y": y, "w": ww, "h": hh})

    if not lines:
        # 没有中文，直接拷贝
        cv2.imwrite(out_path, img_bgr)
        return

    # 为所有行做一个 inpaint 掩模
    mask = np.zeros((h, w), dtype=np.uint8)
    line_infos = []

    for key, words in lines.items():
        # 该行所有词的外接框
        xs = [d["x"] for d in words]
        ys = [d["y"] for d in words]
        xe = [d["x"] + d["w"] for d in words]
        ye = [d["y"] + d["h"] for d in words]
        x1, y1, x2, y2 = min(xs), min(ys), max(xe), max(ye)
        # 适当膨胀一点，避免残留（按行高的 12%）
        margin = int(0.12 * (y2 - y1))
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)

        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

        # 该行的原始文本（按 x 排序）
        words_sorted = sorted(words, key=lambda d: d["x"])
        cn_line = "".join([d["text"] for d in words_sorted])
        line_infos.append({
            "bbox": (x1, y1, x2, y2),
            "text": cn_line
        })

    # 先 inpaint 擦除中文
    # TELEA 方法对自然背景比较柔和
    cleaned = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)

    # 再绘制翻译
    pil_img = Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 字体：若未指定就用系统黑体
    if font_path and os.path.exists(font_path):
        base_font_path = font_path
    else:
        base_font_path = "C:/Windows/Fonts/arialbd.ttf"

    for info in line_infos:
        (x1, y1, x2, y2) = info["bbox"]
        src = info["text"]

        # 翻译（中文->马来语）
        try:
            ms = translator(src)[0]["translation_text"]
        except Exception as e:
            ms = src  # 失败就保留中文，避免空白
            print(f"[警告] 翻译失败：{e}")

        # 从这块区域估计原文字色
        bgr_patch = cleaned[y1:y2, x1:x2]
        rgb_color = estimate_text_color(bgr_patch)

        # 目标行高 & 初始字号
        box_w = x2 - x1
        box_h = y2 - y1
        # 字号略小于行高，给行间距留点余量
        font_size = max(12, int(box_h * 0.86))
        font = ImageFont.truetype(base_font_path, font_size)

        # 文本在框内自动换行：先尝试一行；太宽再按宽度切分
        line = ms.replace("\n", " ").strip()
        if draw.textlength(line, font=font) <= box_w:
            render_lines = [line]
        else:
            # 多行切分直到所有行都能放下
            while True:
                render_lines = wrap_to_width(draw, line, font, box_w)
                total_h = len(render_lines) * font_size + max(0, len(render_lines) - 1) * int(0.2 * font_size)
                if total_h <= box_h or font_size <= 12:
                    break
                font_size = max(12, font_size - 1)
                font = ImageFont.truetype(base_font_path, font_size)

        # 纵向居中，横向默认左对齐（如果原行特别宽，改为中对齐）
        total_h = len(render_lines) * font_size + max(0, len(render_lines) - 1) * int(0.2 * font_size)
        start_y = y1 + max(0, (box_h - total_h) // 2)

        center_align = box_w >= 8 * font_size  # 简单启发
        for idx, ln in enumerate(render_lines):
            text_w = draw.textlength(ln, font=font)
            if center_align:
                sx = x1 + (box_w - text_w) // 2
            else:
                sx = x1

            sy = start_y + idx * (font_size + int(0.2 * font_size))
            draw.text((sx, sy), ln, font=font, fill=rgb_color)

    # 保存
    out_bgr = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, out_bgr)
    print(f"[OK] {os.path.basename(img_path)} -> {out_path}")

# ---------- 主程序 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="input_images", help="输入图片目录")
    ap.add_argument("--output", default="output_images", help="输出图片目录")
    ap.add_argument("--tesseract", default="", help="tesseract.exe 的完整路径（可选）")
    ap.add_argument("--font", default="", help="用于绘制马来语的字体路径（建议粗黑体）")
    args = ap.parse_args()

    # 翻译管线（CPU/GPU自动选择）
    device = 0 if torch.cuda.is_available() else -1
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-ms", device=device)

    in_dir = args.input
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    for name in os.listdir(in_dir):
        if not name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        process_one(
            os.path.join(in_dir, name),
            os.path.join(out_dir, name),
            tesseract_cmd=args.tesseract if args.tesseract else None,
            font_path=args.font if args.font else None,
            translator=translator
        )

if __name__ == "__main__":
    main()
