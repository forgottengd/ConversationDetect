import base64
import os
from typing import Dict, List, Tuple

import numpy as np
import requests
import streamlit as st
import re

import tesserocr
from PIL import Image
import time
from utils import draw_rectangles_yandex, elapsed_time, get_coords_yandex, draw_rectangles
import pytesseract
import cv2

#OCR_API = os.environ['OCR_API']
OCR_API = ""
folder_id = "b1g79vtsge6ggong8ioa"
block_percentile = 0.19  # зона в которую должен входить блок, чтобы считать его за переписку


def encode_file_to_base64(file) -> str:
    """ Кодирует файл изображения в строку base64. """
    encoded_content = base64.b64encode(file)
    return encoded_content.decode('utf-8')


def send_ocr_request_yandex(iam_token, encoded_image) -> str:
    """ Отправляет OCR запрос и возвращает результат в json формате. """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {iam_token}",
        "x-folder-id": folder_id,
        "x-data-logging-enabled": "true"
    }

    body = {
        "mimeType": "JPEG",
        "languageCodes": ["*"],
        "model": "page",
        "content": encoded_image
    }

    response = requests.post("https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText", headers=headers, json=body)
    return response.json()


def process_ocr_yandex(ocr_text, temp) -> Tuple[str, dict]:
    """ Отправляет GPT запрос и возвращает результат в json формате. """
    timestamp_pattern = re.compile(r'^([0-9]|1[0-9]|2[0-3]):([0-5][0-9])$')
    full_text = ""
    result_dict = {
        "sentences": []  # обработанные предложения
    }

    if 'result' in ocr_text:
        if 'textAnnotation' in ocr_text['result']:
            text_annotation = ocr_text['result']['textAnnotation']
            full_text = text_annotation['fullText'] if 'fullText' in text_annotation else ""
            # нет текста - выходим
            if full_text == "":
                full_text = "There is no conversation on image"
                return full_text, result_dict
            result_dict["image_width"] = int(text_annotation["width"])
            result_dict["image_height"] = int(text_annotation["height"])
            result_dict["blocks_overall"] = len(text_annotation['blocks'])
            # храним данные предыдущего сообщения
            current_sentence = {
                "text": "",
                "coords": [0, 0, 0, 0],  # x1, y1, x2, y2
                "side": ""  # "response", "user" or "middle"
            }
            for block in text_annotation['blocks']:
                # если в блоке одна строка со временем, считаем её за временную отметку сообщения
                is_time_block = len(block['lines']) == 1 and timestamp_pattern.match(
                    block['lines'][0]['text']) is not None
                # смотрим насколько далеко был предыдущий блок (сравниваем нижнюю y-координату предыдущего
                # блока с верхней y-координатой текущего)
                # если блок расположен слишком близко, он может относиться
                # к предыдущей строке (или быть временной отметкой)
                current_y = int(block['boundingBox']['vertices'][0]['y'])  # текущая верхняя y-координата
                previous_block_too_close = current_y < current_sentence["coords"][3] + result_dict[
                    "image_height"] * 0.02
                # если верхний левый угол блока находится слева, то предполагаем что это ответ на сообщение
                if int(block['boundingBox']['vertices'][0]['x']) < result_dict["image_width"] * block_percentile:
                    # если текущее предложение отсутствует
                    if current_sentence["text"] == "":
                        current_sentence["coords"] = get_coords_yandex(block)
                        current_sentence['side'] = "response"
                    else:
                        # если блок расположен близко к предыдущему, считаем что он ему принадлежит
                        if previous_block_too_close:
                            current_sentence["coords"][2] = max(int(block['boundingBox']['vertices'][2]['x']),
                                                                current_sentence["coords"][2])
                            current_sentence["coords"][3] = max(int(block['boundingBox']['vertices'][2]['y']),
                                                                current_sentence["coords"][3])
                        # если это новый блок, добавляем предыдущее сообщение и создаем новое
                        else:
                            if current_sentence['side'] != "middle":
                                result_dict["sentences"].append(current_sentence.copy())
                            current_sentence["text"] = ""
                            current_sentence["coords"] = get_coords_yandex(block)
                            current_sentence['side'] = "response"
                    for line in block["lines"]:
                        current_sentence["text"] += f"{line['text']} "
                # или если верхний правый угол блока находится справа, то предполагаем что это сообщение пользователя
                elif int(block['boundingBox']['vertices'][3]['x']) > result_dict["image_width"] * (
                        1 - block_percentile):
                    # если текущее предложение отсутствует
                    if current_sentence["text"] == "":
                        current_sentence["coords"] = get_coords_yandex(block)
                        current_sentence['side'] = "user"
                    else:
                        # если блок расположен близко к предыдущему, считаем что он ему принадлежит
                        if previous_block_too_close:
                            # расширяем х- и y-координаты влево, вправо и вниз
                            current_sentence["coords"][0] = min(int(block['boundingBox']['vertices'][0]['x']),
                                                                current_sentence["coords"][0])
                            current_sentence["coords"][1] = min(int(block['boundingBox']['vertices'][0]['y']),
                                                                current_sentence["coords"][1])
                            current_sentence["coords"][2] = max(int(block['boundingBox']['vertices'][2]['x']),
                                                                current_sentence["coords"][2])
                            current_sentence["coords"][3] = max(int(block['boundingBox']['vertices'][2]['y']),
                                                                current_sentence["coords"][3])
                            # проверяем на сторону, возможно предыдущий блок был слишком близко расположен к процентилю
                            # границы и неправильно записан в response. проверяем по block_percentile / 2
                            # если ближе к правой границе чем к левой то меняем сторону
                            if (current_sentence["coords"][2] >= result_dict["image_width"] * (
                                    1 - block_percentile / 2) and
                                    current_sentence["coords"][0] >= result_dict["image_width"] * (
                                            block_percentile / 2)):
                                current_sentence['side'] = "user"
                        else:
                            result_dict["sentences"].append(current_sentence.copy())
                            current_sentence["text"] = ""
                            current_sentence["coords"] = get_coords_yandex(block)
                            current_sentence['side'] = "user"
                    for line in block["lines"]:
                        current_sentence["text"] += f"{line['text']} "
                # если блок не входит в процентную зону краев картинки
                else:
                    # если блок расположен близко к предыдущему, считаем что он ему принадлежит
                    if previous_block_too_close:
                        # расширяем границы текущего блока влево, вправо и вниз
                        current_sentence["coords"][0] = min(int(block['boundingBox']['vertices'][0]['x']),
                                                            current_sentence["coords"][0])
                        current_sentence["coords"][2] = max(int(block['boundingBox']['vertices'][2]['x']),
                                                            current_sentence["coords"][2])
                        current_sentence["coords"][3] = max(int(block['boundingBox']['vertices'][2]['y']),
                                                            current_sentence["coords"][3])
                        for line in block["lines"]:
                            current_sentence["text"] += f"{line['text']} "
                    else:
                        # если предыдущий блок не центральный, а относится к какой-то части, то добавляем его
                        if current_sentence['side'] != "middle":
                            result_dict["sentences"].append(current_sentence.copy())
                            current_sentence["text"] = ""
                            current_sentence["coords"] = get_coords_yandex(block)
                            current_sentence['side'] = "middle"
                            for line in block["lines"]:
                                current_sentence["text"] += f"{line['text']} "
            result_dict["sentences"].append(current_sentence)
    return full_text, result_dict


def process_dict_yandex(ocr_dict: Dict):
    def check_alternation(messages):
        if len(messages) < 2:
            return False, 0
        last_side = messages["sentences"][0]["side"]
        swap = 0
        for message in messages["sentences"][1:]:
            if message["side"] != last_side:
                last_side = message["side"]
                swap += 1
        return swap > 1, swap

    alteration, swap_count = check_alternation(ocr_dict)
    messages_count = len(ocr_dict["sentences"])
    return alteration, swap_count, messages_count


def on_click_yandex(file, temp):
    """ Вызывает обработку загруженного изображения и отображает результат """
    with st.spinner("Detecting image..."):
        try:
            start_time = time.time()  # засекаем время
            image = encode_file_to_base64(file.getvalue())
            ocr_response = send_ocr_request_yandex(OCR_API, image)
            print(f"ocr_response обработан за {elapsed_time(start_time, time.time())}")
            # parse OCR result
            #ocr_response = res.res2
            if 'error' in ocr_response:
                raise Exception(ocr_response['error'])
            full_text, result_dict = process_ocr_yandex(ocr_response, temp)
            metrics = process_dict_yandex(result_dict)
            confidence = min(metrics[0] * 0.25 + metrics[1] * 0.04 + metrics[2] * 0.08, 1.0)
            print(f"2 стороны: {metrics[0]}, чередования сторон: {metrics[1]}, количество сообщений: {metrics[2]}")
            decision = "Переписка" if confidence >= temp else "Не переписка"
            decision += f". Уверенность: {confidence}. Время выполнения: {elapsed_time(start_time, time.time())}"
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.text_area(label="Result", value=decision, height=40)
    col1, col2 = st.columns([1, 1], gap='medium')
    with col1:
        st.image(file, caption="Загруженное изображение")
    with col2:
        st.image(draw_rectangles_yandex(file,
                                        ocr_response['result']['textAnnotation']['blocks'],
                                        result_dict['sentences']),
                 caption="Обработанное изображение", use_column_width=True)


def main():
    """ Создает объекты streamlit-сервиса """
    uploader = st.file_uploader("Choose image for detection", type=['png', 'jpg', 'jpeg'])
    slider_value = st.empty()
    show_slider = st.checkbox("Настроить порог")
    if show_slider:
        slider_value = st.slider(
            'Threshold',
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            value=0.8
        )
    if uploader:
        temp = slider_value if isinstance(slider_value, float) else 0.8
        # on_click_yandex(uploader, temp)
        on_upload(uploader, temp)


def on_upload(file, temp: float):
    """ Вызывает обработку загруженного изображения и отображает результат """
    with st.spinner("Detecting image..."):
        try:
            start_time = time.time()  # засекаем время
            # get OCR result
            ocr_result, image_width, image_height = parse_image(file)
            print(f"parse_image обработан за {elapsed_time(start_time, time.time())}")
            processed_result = process_image(ocr_result, image_width, image_height)
            print(f"process_image обработан за {elapsed_time(start_time, time.time())}")
            #print(processed_result['text_blocks'])
            metrics = parse_ocr(processed_result)
            confidence = round(min(metrics[0] * 0.3 + metrics[1] * 0.05 + processed_result["blocks_overall"] * 0.07,
                                   1.0), 2)
            decision = "Переписка" if confidence >= temp else "Не переписка"
            decision += f". Уверенность: {confidence}. Время выполнения: {elapsed_time(start_time, time.time())}"
        except Exception as e:
            print(f"Error: {e}")
            st.error(f"Error: {e}")
            st.stop()

    st.text_area(label="Result", value=decision, height=40)
    col1, col2 = st.columns([1, 1], gap='medium')
    with col1:
        st.image(file, caption="Загруженное изображение")
    with col2:
        st.image(draw_rectangles(file, ocr_result, processed_result['text_blocks']),
                 caption="Обработанное изображение", use_column_width=True)


def process_image(details, image_width, image_height) -> Dict:
    text_blocks = []  # Список для хранения информации о текстовых блоках
    reduce_factor = 1.3  # коэффициент для уменьшения границ для обнаружения широких блоков
    full_text = ""  # тут храним весь найденный текст (для чего? кек)
    current_message = {  # храним данные текущего сообщения
        "text": "",
        "coords": [0, 0, 0, -100],  # x1, y1, x2, y2
        "side": ""  # "response", "user" or "middle"
    }
    for i in range(len(details['text'])):
        if details['text'][i].strip() != '':
            full_text += f"{details['text'][i]} "
            # смотрим насколько далеко был предыдущий блок (сравниваем нижнюю y-координату предыдущего
            # блока с верхней y-координатой текущего), если блок расположен слишком близко, он может относиться
            # к предыдущей строке (или быть временной отметкой)
            current_y = details['top'][i]  # текущая верхняя y-координата
            previous_block_too_close = current_y < current_message["coords"][3] + image_height * 0.02
            # если верхний левый угол блока находится слева, то предполагаем что это ответ на сообщение
            if details['left'][i] < image_width * block_percentile:
                # если текущее предложение отсутствует
                if current_message["text"] == "":
                    current_message["coords"] = [details['left'][i], details['top'][i],
                                                 details['left'][i] + details['width'][i],
                                                 details['top'][i] + details['height'][i]]
                    current_message['side'] = "response"
                else:
                    # если блок расположен близко к предыдущему, считаем что он ему принадлежит
                    if previous_block_too_close:
                        current_message["coords"][0] = min(details['left'][i],
                                                           current_message["coords"][0])
                        current_message["coords"][2] = max(details['left'][i] + details['width'][i],
                                                           current_message["coords"][2])
                        current_message["coords"][3] = max(details['top'][i] + details['height'][i],
                                                           current_message["coords"][3])
                        if (current_message["coords"][2] <= image_width * (1 - block_percentile) and
                                current_message["coords"][0] <= image_width * block_percentile):
                            current_message['side'] = "response"
                    # если это новый блок, добавляем предыдущее сообщение и создаем новое
                    else:
                        if not (current_message["coords"][2] >= image_width * (1 - block_percentile / reduce_factor) and
                                current_message["coords"][0] <= image_width * (block_percentile / reduce_factor)):
                            text_blocks.append(current_message.copy())
                        current_message["text"] = ""
                        current_message["coords"] = [details['left'][i], details['top'][i],
                                                     details['left'][i] + details['width'][i],
                                                     details['top'][i] + details['height'][i]]
                        current_message['side'] = "response"
                current_message["text"] += f"{details['text'][i]} "
            # или если верхний правый угол блока находится справа, то предполагаем что это сообщение пользователя
            elif details['left'][i] + details['width'][i] > image_width * (1 - block_percentile):
                # если текущее сообщение отсутствует
                if current_message["text"] == "":
                    current_message["coords"] = [details['left'][i], details['top'][i],
                                                 details['left'][i] + details['width'][i],
                                                 details['top'][i] + details['height'][i]]
                    current_message['side'] = "user"
                else:
                    # если блок расположен близко к предыдущему, считаем что он ему принадлежит
                    if previous_block_too_close:
                        # расширяем х- и y-координаты влево, вправо и вниз
                        current_message["coords"][0] = min(details['left'][i],
                                                           current_message["coords"][0])
                        current_message["coords"][1] = min(details['top'][i],
                                                           current_message["coords"][1])
                        current_message["coords"][2] = max(details['left'][i] + details['width'][i],
                                                           current_message["coords"][2])
                        current_message["coords"][3] = max(details['top'][i] + details['height'][i],
                                                           current_message["coords"][3])
                        # проверяем на сторону, возможно предыдущий блок был слишком близко расположен к процентилю
                        # границы и неправильно записан в response. проверяем по block_percentile, если ближе к правой
                        # границе чем к левой то меняем сторону. если сторона=response, то не сменится на user
                        if (current_message["coords"][2] >= image_width * (1 - block_percentile) and
                                current_message["coords"][0] >= image_width * block_percentile):
                            current_message['side'] = "user"
                    else:
                        if not (current_message["coords"][2] >= image_width * (1 - block_percentile / reduce_factor) and
                                current_message["coords"][0] <= image_width * (block_percentile / reduce_factor)):
                            text_blocks.append(current_message.copy())
                        current_message["text"] = ""
                        current_message["coords"] = [details['left'][i], details['top'][i],
                                                     details['left'][i] + details['width'][i],
                                                     details['top'][i] + details['height'][i]]
                        current_message['side'] = "user"
                current_message["text"] += f"{details['text'][i]} "
            # если блок не входит в процентную зону краев картинки
            else:
                # если блок расположен близко к предыдущему, считаем что он ему принадлежит
                if previous_block_too_close:
                    # расширяем границы текущего блока влево, вправо и вниз
                    current_message["coords"][0] = min(details['left'][i],
                                                       current_message["coords"][0])
                    current_message["coords"][2] = max(details['left'][i] + details['width'][i],
                                                       current_message["coords"][2])
                    current_message["coords"][3] = max(details['top'][i] + details['height'][i],
                                                       current_message["coords"][3])
                else:
                    # если предыдущий блок не центральный, а относится к какой-то части, то добавляем его
                    if (current_message["text"] != "" and not
                       (current_message["coords"][2] >= image_width * (1 - block_percentile / reduce_factor) and
                       current_message["coords"][0] <= image_width * (block_percentile / reduce_factor))):
                        text_blocks.append(current_message.copy())
                    current_message["text"] = ""
                    current_message["coords"] = [details['left'][i], details['top'][i],
                                                 details['left'][i] + details['width'][i],
                                                 details['top'][i] + details['height'][i]]
                    current_message['side'] = "middle"
                current_message["text"] += f"{details['text'][i]} "

    if (current_message["text"] != "" and not
       (current_message["coords"][2] >= image_width * (1 - block_percentile / reduce_factor) and
       current_message["coords"][0] <= image_width * (block_percentile / reduce_factor))):
        text_blocks.append(current_message)
    result_dict = {
        "blocks_overall": len(text_blocks),
        "text_blocks": text_blocks,
        "full_text": full_text
    }
    return result_dict


def parse_ocr(processed_result):
    def check_alternation(messages):
        if len(messages["text_blocks"]) < 2:
            return False, 0
        last_side = messages["text_blocks"][0]["side"]
        swap = 0
        for message in messages["text_blocks"][1:]:
            if message["side"] != last_side:
                last_side = message["side"]
                swap += 1
        return swap > 1, swap

    alteration, swap_count = check_alternation(processed_result)
    return alteration, swap_count


def parse_image(file):
    # Загрузка изображения
    #image = cv2.imread(file)
    #pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Пример для Windows
    # Читаем файл как массив байт
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    # Перевод в серый для улучшения распознавания
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Используем Tesseract для распознавания текста и получения детальной информации о расположении текста
    details = pytesseract.image_to_data(gray_image, lang='eng+rus', config=r'--oem 3 --psm 4',
                                        output_type=pytesseract.Output.DICT)
    return details, image.shape[1], image.shape[0]


if __name__ == "__main__":
    #on_upload("C:\\Users\\DragonsHome\\Desktop\\test\\9.jpg", 0.8)
    main()
