
# 🖼️ Chinese-to-Malay Image Translator

This project is designed to automatically translate Chinese text in images into **Malay**, while preserving the **layout, style, font size, color, and position** of the original image.

> ✅ Ideal for e-commerce image localization, especially for platforms like TikTok Shop, Shopee, and Lazada.

---

## 📌 Features

- ✅ Detects Chinese text in images using OCR
- ✅ Translates Chinese → Malay using transformer-based models
- ✅ Automatically redraws text in original style on the image
- ✅ Batch processing of multiple images
- ✅ Fully offline-capable (optional)

---

## 📁 Directory Structure

```
project-root/
├── input_images/         # Folder for original Chinese-language images
├── output_images/        # Folder where translated images will be saved
├── fonts/                # (Optional) Custom fonts for accurate rendering
├── translate_images.py   # Main script: OCR + Translate + Redraw
├── requirements.txt      # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/image-zh2ms-translator.git
cd image-zh2ms-translator
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> Requirements include: `pytesseract`, `transformers`, `torch`, `Pillow`, `opencv-python`

### 3. Install Tesseract OCR

Make sure Tesseract is installed with Simplified Chinese language support (`chi_sim`):

- **Ubuntu:**
  ```bash
  sudo apt install tesseract-ocr tesseract-ocr-chi-sim
  ```
- **macOS (Homebrew):**
  ```bash
  brew install tesseract
  brew install tesseract-lang
  ```
- **Windows:**  
  Download from: https://github.com/tesseract-ocr/tesseract

---

## 🧠 How It Works

1. **OCR Module**: Extracts Chinese text from the image using Tesseract.
2. **Translation Module**: Translates the extracted Chinese text to Malay using pretrained models from HuggingFace.
3. **Drawing Module**: Replaces the Chinese text in the image with translated Malay text using Pillow/OpenCV, preserving style.

---

## 🖼️ Example

### Input:

A promotional image with Chinese:
```
- 强悍吸力
- 承重力强
- 稳固晾晒
- 轻松承重四季衣物 一架搞定
```

### Output:

Translated into Malay:
```
- Sedutan kuat
- Kapasiti galas beban yang kuat
- Pengeringan tegas
- Satu rak boleh membawa semua empat musim pakaian dengan mudah
```

With original visual layout preserved.

---

## 🛠️ Usage

### Run the script

```bash
python translate_images.py
```

All images in `input_images/` will be processed and output to `output_images/`.

---

## 📚 Customization

- Modify `translator.py` to use your own translation dictionary or API (e.g., Google Translate API).
- Adjust `drawer.py` to customize font, color, spacing, or fallback rendering logic.

---

## 🤖 Codex/AI Autogeneration Instructions

> This repository is structured to support Copilot/Codex to:
> - Autocomplete Python modules like `ocr.py`, `translator.py`, `drawer.py`
> - Use docstrings and function naming to infer logic
> - Suggest improvements or modularize components automatically

If you're using Copilot or ChatGPT Code Interpreter, prompt:
```
Generate translate_images.py using Tesseract OCR to detect Chinese text in images and translate to Malay using HuggingFace transformers, then redraw on the image using Pillow.
```

---

## 💬 License

MIT License

---

## 🙋‍♂️ Contact

For commercial solutions (bulk image translation, OCR + localization AI tools), please reach out via GitHub Issues or email.
