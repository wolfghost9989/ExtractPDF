import json
import os
import time
import re
from functools import wraps
from logging.handlers import RotatingFileHandler

from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont

import logging
from unidecode import unidecode
import datetime
import logging

log_format = "%(asctime)s::%(levelname)s::%(name)s::" \
             "%(filename)s::%(lineno)d::%(message)s"
logger = logging.getLogger(__name__)

# To override the default severity of logging
logger.setLevel('DEBUG')

# Use FileHandler() to log to a file
file_handler = RotatingFileHandler("log/ocr_document.log",
                                 # mode='a',
                                 maxBytes=5 * 1024 * 1024,
                                 backupCount=2,
                                 encoding=None)
formatter = logging.Formatter(log_format)
file_handler.setFormatter(formatter)

# Don't forget to add the file handler
logger.addHandler(file_handler)

# Draw text box and text in image
def draw_result(dt_boxes, text_array, image, font_path):
    for index, box in enumerate(dt_boxes):
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(image, [box], True, color=(255, 255, 0), thickness=1)

    # Convert to Image to draw text vietnamese
    image = Image.fromarray(image)
    img_new = image.copy()
    draw = ImageDraw.Draw(img_new)
    color = (0, 0, 255)
    # font_path = "/home/tms/Documents/TaiDV/projects/ocr/ocr_framework/PaddleOCR/doc/fonts/latin.ttf"
    # font_path = "/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-B.ttf"
    # font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    for index, box in enumerate(dt_boxes):
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        text = text_array[index]
        scale = 1  # this value can be from 0 to 1 (0,1] to change the size of the text relative to the image
        imageWidth = max((box[1][0] - box[0][0]), (box[2][0] - box[3][0]))
        imageHeight = max((box[3][1] - box[0][1]), (box[2][1] - box[1][1]))
        fontScale = min(imageWidth, imageHeight) / (1.5 / scale)
        font = ImageFont.truetype(font_path, int(fontScale), encoding='utf-8')
        if index == 0:
            box[3][0] = 0
        draw.text((box[3][0], box[3][1]), text, fill=color, font=font)

    img_new = Image.blend(image, img_new, 0.5)
    return np.array(img_new)


# Crop image form textbox
def crop_image(image, point):
    top_left_x = int(min([point[0][0], point[1][0], point[2][0], point[3][0]]))
    top_left_y = int(min([point[0][1], point[1][1], point[2][1], point[3][1]]))
    bot_right_x = int(max([point[0][0], point[1][0], point[2][0], point[3][0]]))
    bot_right_y = int(max([point[0][1], point[1][1], point[2][1], point[3][1]]))

    return image[max(top_left_y - int((bot_right_y - top_left_y) / 7), 0): \
                 max(bot_right_y + int((bot_right_y - top_left_y) / 7), 0), \
           top_left_x: bot_right_x]


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


# Load image from file (jpg, pdf)
def load_array_image_from_file(path_file):
    filename = os.path.split(path_file)[-1]
    filename_split = filename.split(".")
    # check file pdf and retrive first page
    if filename_split[-1] == "pdf":
        images = convert_from_path(path_file)
        if len(images) == 0:
            logger.info("Error in loading pdf: {}".format(path_file))
        image = images[0]
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:  # image
        image = cv2.imread(path_file)
        if image is None:
            logger.info("Error in loading image: {}".format(path_file))
        else:
            logger.info("Loaded image: {}".format(path_file))

    return image


def save_image_postprocess(result_boxes, result_array_text, image,
                           path_font_show, path_file, path_folder_output):
    image_drawed = draw_result(result_boxes, result_array_text,
                               image, path_font_show)
    filename = os.path.split(path_file)[-1]
    filename_split = filename.split(".")
    if filename_split[-1] == "pdf":
        filename_split[-1] = "jpg"
        filename = ".".join(filename_split)
    path_file = os.path.join(path_folder_output,
                             "processed_{}".format(filename))
    cv2.imwrite(path_file, image_drawed)
    logger.info("The visualized image saved in {}".format(path_file))


# Find answer for text box based on check question in position right on a line
def check_line_answer_right(list_text_sorted, list_box_sorted, index):
    list_text = []
    list_box = []
    # Select 2 text box top_left_y sorted above and bottom
    y_min = min(list_box_sorted[index][i][1] for i in range(4))
    y_max = max(list_box_sorted[index][i][1] for i in range(4))
    len_list_text = len(list_text_sorted)
    if index > 2 and index < (len_list_text - 2):
        index_check = [index - 2, index - 1, index + 1, index + 2]
        for i in index_check:
            # Check point between [y_min, y_max]
            box_temp = list_box_sorted[i]
            y_between = (min(box_temp[i][1] for i in range(4)) +
                         max(box_temp[i][1] for i in range(4))) / 2

            if y_between < y_max and y_between > y_min:
                list_text.append(list_text_sorted[i])
                list_box.append(list_box_sorted[i])

        return list_text, list_box
    else:
        return [], []


# import the necessary packages
import numpy as np
import cv2


def margin_pst(point):
    # top_left_x = int(min([point[0][0], point[1][0], point[2][0], point[3][0]]))
    top_left_y = int(min([point[0][1], point[1][1], point[2][1], point[3][1]]))
    # bot_right_x = int(max([point[0][0], point[1][0], point[2][0], point[3][0]]))
    bot_right_y = int(max([point[0][1], point[1][1], point[2][1], point[3][1]]))

    margin = int((bot_right_y - top_left_y) / 7)
    point[0][0] = point[0][0] - margin
    point[0][1] = point[0][1] - margin
    point[1][0] = point[1][0] + margin
    point[1][1] = point[1][1] - margin
    point[2][0] = point[2][0] + margin
    point[2][1] = point[2][1] + margin
    point[3][0] = point[3][0] - margin
    point[3][1] = point[3][1] + margin

    return point


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def rotate_image(image, result_boxes):
    # Find index line with maximum width
    width_max = 0
    index_width_max = None
    for index, box in enumerate(result_boxes):
        left_top_point = box[0]
        right_bottom_point = box[2]

        width_temp = right_bottom_point[0] - left_top_point[0]
        if width_temp > width_max:
            width_max = width_temp
            index_width_max = index

    if index_width_max == None:
        logger.info("Error value of result_boxes")
        return image

    else:
        angle = cv2.minAreaRect(result_boxes[index_width_max])[-1]
        if angle < 45:
            angle = angle
        else:
            angle = angle - 90
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)
        return rotated_image


def rotate_image(image, result_boxes):
    # Find index line with maximum width
    width_max = 0
    index_width_max = None
    for index, box in enumerate(result_boxes):
        left_top_point = box[0]
        right_bottom_point = box[2]
        width_temp = right_bottom_point[0] - left_top_point[0]
        if width_temp > width_max:
            width_max = width_temp
            index_width_max = index
    if index_width_max == None:
        logger.info("Error value of result_boxes")
        return image, None
    else:
        angle = cv2.minAreaRect(result_boxes[index_width_max])[-1]
        if angle < 45:
            angle = angle
        else:
            angle = angle - 90
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)
        return rotated_image, angle


def matching_box(result_boxes, result_texts, index_title, image):
    result_boxes_sorted = []
    result_texts_sorted = []
    (height, width, _) = image.shape
    for index, box in enumerate(result_boxes):
        left_top_point = box[0]
        left_bottom_point = box[3]

        # Match box have index title > 2
        if index < index_title + 2:
            result_texts_sorted.append(result_texts[index])
            result_boxes_sorted.append(result_boxes[index])
            continue
        # Match box have index title > 2 and smaller len box because full stack
        if left_top_point[0] < width / 4 and index < len(result_boxes) - 2:
            y_start_line = left_top_point[1]
            y_end_line = left_bottom_point[1]
            # Select 2 box above and 2 box below
            index_check = [index - 2, index - 1, index + 1, index + 2]
            box_ok = []  # box satisfy
            text_ok = []  # text satisfy
            for i in index_check:
                y_between = (min(result_boxes[i][j][1] for j in range(4)) +
                             max(result_boxes[i][j][1] for j in range(4))) / 2
                if y_between < y_end_line and y_between > y_start_line:
                    box_ok.append(result_boxes[i])
                    text_ok.append(result_texts[i])
            # No box satisfy
            if len(box_ok) == 0:
                result_texts_sorted.append(result_texts[index])
                result_boxes_sorted.append(result_boxes[index])
                continue
            # 1 box satisfy
            elif len(box_ok) == 1:
                if box_ok[0][0][0] >= left_top_point[0]:  # Element first in list box_ok
                    text_link = result_texts[index] + " " + text_ok[0]
                    result_texts_sorted.append(text_link)
                else:
                    result_texts_sorted.append(result_texts[index])
                    # result_boxes_sorted.append(result_boxes[index])
                    continue
            # Multi box satisfy -> select box nearest with box processing
            else:
                index_box = None
                distance_min = 10000
                for index_box_temp, box in enumerate(box_ok):
                    temp = box[0][0] - left_top_point[0]
                    if (temp > 0) and (temp < distance_min) and (box[0][0] < width / 4 * 3):
                        distance_min = temp
                        index_box = index_box_temp

                # Exist box satisfy distance min
                if index_box != None:
                    text_link = result_texts[index] + " " + text_ok[index_box]
                    result_texts_sorted.append(text_link)
                # No Exist box satisfy
                else:
                    result_texts_sorted.append(result_texts[index])
        else:
            result_texts_sorted.append(result_texts[index])

    return result_texts_sorted

def recognize_text(result_boxes, image, recognize):
    # Recognize image -> text in ever text_box
    result_texts = []
    for index, point in enumerate(result_boxes):
        point = margin_pst(point)
        # Transform for text not tilted
        image_ocr = four_point_transform(image, point)
        image_ocr = cv2.cvtColor(image_ocr, cv2.COLOR_BGR2RGB)
        image_ocr = Image.fromarray(image_ocr)
        result_text = recognize.predict(image_ocr)
        result_texts.append(result_text)

    # Reverse of texts, boxes
    result_texts.reverse()
    result_boxes = result_boxes[::-1]

    return result_texts, result_boxes

def recognize_text_title(result_boxes, image, recognize,
                         list_title_contracts, list_config_contracts):
    # Recognize image -> text in ever text_box
    result_texts = []
    for index, point in enumerate(result_boxes[::-1]):
        point = margin_pst(point)
        # Transform for text not tilted
        image_ocr = four_point_transform(image, point)
        image_ocr = cv2.cvtColor(image_ocr, cv2.COLOR_BGR2RGB)
        image_ocr = Image.fromarray(image_ocr)
        result_text = recognize.predict(image_ocr)
        # Because result_text is title:
        # Check length of line > 50 and
        # count character upper in this line < len(character in line) /2
        # -> page not first page
        len_result_text = len(result_text)
        if len_result_text > 50 and count_upper(result_text) < (len_result_text / 2):
            return False
        text_box_lower_unidecode = unidecode(result_text)
        for index_contract, title in enumerate(list_title_contracts):
            # Replace character special = "", because detect . , ...
            # Case detect missing character special will error
            text_box_lower_unidecode = re.sub('[^A-Za-z0-9& ]+', '',
                                              text_box_lower_unidecode)
            if re.search(text_box_lower_unidecode, title, re.IGNORECASE) != None and \
                len(text_box_lower_unidecode) > (len(title) / 4 * 3):
                logger.info(text_box_lower_unidecode)
                return True
    return False

def replace_spell(text_test):
    words = [
        
    ]

    for word in words:
        word_lower_unidecode = unidecode(word.lower())
        result_search = re.search(word_lower_unidecode, unidecode(text_test), re.IGNORECASE)
        if result_search is not None:
            (start_point, end_point) = result_search.span()
            text_test = text_test.replace(text_test[start_point:end_point], word)

    return text_test


def count_upper(str):
    # upper, lower, number, special = 0, 0, 0, 0
    upper = 0
    for i in range(len(str)):
        if str[i].isupper():
            upper += 1
        # elif str[i].islower():
        #     lower += 1
        # elif str[i].isdigit():
        #     number += 1
        # else:
        #     special += 1

    return upper

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

