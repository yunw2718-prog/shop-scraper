import argparse
import os
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pytesseract

# Regular expression to check for Chinese characters
CJK_RE = re.compile(r'[\u4e00-\u9fff]')

# Simple offline Chinese to Malay dictionary
CHINESE_TO_MALAY = {
    "你好": "halo",
    "谢谢": "terima kasih",
    "重量": "berat",
    "价格": "harga",
    "长度": "panjang",
}

def translate_to_malay(text: str) -> str:
    """Translate Chinese text to Malay using an offline dictionary.

    If a full phrase is not found in the dictionary, the function falls back to
    character level translation. Unknown characters/words are returned as-is.
    """
    if text in CHINESE_TO_MALAY:
        return CHINESE_TO_MALAY[text]
    # fallback to character by character translation
    return ''.join(CHINESE_TO_MALAY.get(ch, ch) for ch in text)

def estimate_text_color(region: Image.Image) -> tuple:
    """Estimate the text color within the given region.

    The function converts the region to a numpy array and selects the median
    color among the darker pixels to approximate the original text color.
    """
    arr = np.array(region)
    if arr.size == 0:
        return (0, 0, 0)
    gray = arr.mean(axis=2)
    thr = np.percentile(gray, 30)
    mask = gray <= thr
    if np.count_nonzero(mask) < 10:
        color = np.median(arr.reshape(-1, 3), axis=0)
    else:
        color = np.median(arr[mask], axis=0)
    return tuple(int(c) for c in color)

def process_image(input_path: str, output_path: str, font_path: str | None = None) -> None:
    """Load an image, translate Chinese text to Malay and render back."""
    image = Image.open(input_path)
    info = image.info  # Preserve metadata
    translated = image.copy()
    draw = ImageDraw.Draw(translated)

    # OCR to extract text and bounding boxes
    data = pytesseract.image_to_data(image, lang="chi_sim", output_type=pytesseract.Output.DICT)
    n = len(data['text'])

    for i in range(n):
        text = data['text'][i].strip()
        if not text:
            continue
        if not CJK_RE.search(text):
            continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        bbox = (x, y, x + w, y + h)

        malay = translate_to_malay(text)

        # Determine font size based on height
        font_size = max(12, int(h * 0.9))
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()

        # Estimate original text color
        region = image.crop(bbox)
        color = estimate_text_color(region)

        # Fill the region with average background color to remove original text
        bg_color = region.resize((1, 1), resample=Image.Resampling.BILINEAR).getpixel((0, 0))
        draw.rectangle(bbox, fill=bg_color)

        # Draw translated text
        draw.text((x, y), malay, fill=color, font=font)

    # Save image with original metadata and resolution
    translated.save(output_path, **info)

def main() -> None:
    parser = argparse.ArgumentParser(description="Translate Chinese text on an image to Malay.")
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument("output", help="Path to save the translated image")
    parser.add_argument("--font", default=None, help="Optional path to a .ttf font file")
    args = parser.parse_args()

    process_image(args.input, args.output, args.font)

if __name__ == "__main__":
    main()
