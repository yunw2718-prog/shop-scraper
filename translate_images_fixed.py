
import os
from PIL import Image, ImageDraw, ImageFont
import easyocr

# 初始化OCR识别器（中英文）
reader = easyocr.Reader(['ch_sim', 'en'])

def translate_text(text):
    # 示例翻译：你可以接入 Google Translate 或其他翻译API
    return text.replace("厘米", "cm").replace("长度", "Length")

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            print(f"正在处理图片: {image_path}")
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)

            results = reader.readtext(image_path)

            for (bbox, text, prob) in results:
                translated = translate_text(text)
                print(f"识别结果: {text} -> 翻译: {translated}")
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))

                # 使用微软雅黑字体，替代 simhei.ttf
                try:
                    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
                except Exception as e:
                    print("字体加载失败，使用默认字体")
                    font = ImageFont.load_default()

                draw.rectangle([top_left, bottom_right], outline="red", width=2)
                draw.text(top_left, translated, fill="blue", font=font)

            output_path = os.path.join(output_folder, filename)
            print(f"保存翻译后图片到: {output_path}")
            image.save(output_path)

if __name__ == "__main__":
    input_folder = "input_images"
    output_folder = "output_images"
    process_images(input_folder, output_folder)
