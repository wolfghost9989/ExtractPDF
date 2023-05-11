import cv2
import sys
import numpy as np
from PIL import Image
import tools.infer.utility as utility
from PyPDF2 import PdfFileWriter, PdfReader
from tools.predict_det import TextDetector
from vietocr.tool.predictor import Predictor
# from vietocr.tool.config import Cfg
from utils import four_point_transform, margin_pst, read_json, draw_result
from config import Cfg
from pdf2image import convert_from_path


def process_image(image, detect, recognize):
    image = np.array(image)

    result_boxes, time_detect = detect(image)
    width_max = 0
    index_width_max = None
    for index, box in enumerate(result_boxes):
        left_top_point = box[0]
        right_bottom_point = box[2]

        width_temp = right_bottom_point[0] - left_top_point[0]
        if width_temp > width_max:
            width_max = width_temp
            index_width_max = index

    angle = cv2.minAreaRect(result_boxes[index_width_max])[-1]
    print(angle)
    if angle < 45:
        angle = angle
    else:
        angle = angle - 90
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
    
    result_boxes, time_detect = detect(rotated_image)
    result_texts = []
    for index, point in enumerate(result_boxes):
        point = margin_pst(point)
        # Transform for text not tilted
        image_ocr = four_point_transform(rotated_image, point)
        image_ocr = cv2.cvtColor(image_ocr, cv2.COLOR_BGR2RGB)
        image_ocr = Image.fromarray(image_ocr)
        result_text = recognize.predict(image_ocr)
        result_texts.append(result_text)

    # Reverse of texts, boxes
    result_texts.reverse()
    result_boxes = result_boxes[::-1]

    return result_texts, result_boxes
