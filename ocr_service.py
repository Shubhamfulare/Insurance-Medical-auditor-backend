import os
from google.cloud import vision
import pytesseract
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR


# vision ocr
def extract_text(image_path: str):
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as img:
        content = img.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if not texts:
        return ""

    return texts[0].description

# switch to tesseract cause of billing issue



# Windows only (skip on Linux/Mac)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_local(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError("Image not found")

    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)

    return text.strip()



# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
# model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten"
)
model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)



def preprocess_line(gray_img):
    """
    Light preprocessing – preserves pen strokes
    """
    gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    gray_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
    return Image.fromarray(gray_img).convert("RGB")

# ================= LINE SEGMENTATION =================
def segment_lines(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Cannot read image")

    # Only for segmentation (not OCR)
    _, thresh = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []
    for cnt in sorted(contours, key=lambda c: cv2.boundingRect(c)[1]):
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 20 and w > 120:
            line = img[y:y+h, x:x+w]
            lines.append(line)

    return lines




# ================= OCR USING TrOCR =================
def extract_text_trocr(image_path):
    lines = segment_lines(image_path)
    all_text = []

    for line in lines:
        line_img = preprocess_line(line)

        pixel_values = processor(
            images=line_img,
            return_tensors="pt"
        ).pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )

        text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

        if len(text) > 2:
            all_text.append(text)

    return "\n".join(all_text)




# Initialize once (printed / structured docs)
paddle_ocr = PaddleOCR(
    lang="en",
    use_angle_cls=True,
    # show_log=False
)

def extract_text_paddle(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError("Image not found")

    result = paddle_ocr.ocr(image_path)

    lines = []

    for line in result:
        for word in line:

            text = None
            conf = None

            # Case 1: [box, (text, conf)]
            if isinstance(word, list) and len(word) == 2:
                if isinstance(word[1], (list, tuple)):
                    text = word[1][0]
                    conf = word[1][1] if len(word[1]) > 1 else None
                elif isinstance(word[1], str):
                    text = word[1]

            # Case 2: plain string
            elif isinstance(word, str):
                text = word

            if text and len(text.strip()) > 1:
                lines.append(text.strip())

    return "\n".join(lines)