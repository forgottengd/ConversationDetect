from typing import List, Tuple, Dict
import re

import numpy as np
import requests


folder_id = "b1g79vtsge6ggong8ioa"


def get_coords_yandex(block) -> List:
    return ([int(block['boundingBox']['vertices'][0]['x']),
            int(block['boundingBox']['vertices'][0]['y']),
            int(block['boundingBox']['vertices'][2]['x']),
            int(block['boundingBox']['vertices'][2]['y'])])


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


def process_ocr_yandex(ocr_text, block_percentile) -> Tuple[str, dict]:
    """ Отправляет GPT запрос и возвращает результат в json формате. """
    reduce_factor = 1.5
    timestamp_pattern = re.compile(r'^([0-9]|1[0-9]|2[0-3]):([0-5][0-9])$')
    full_text = ""
    result_dict = {
        "sentences": []  # обработанные предложения
    }
    # если нет ожидаемой структуры - выходим
    if 'result' not in ocr_text or 'textAnnotation' not in ocr_text['result']:
        return full_text, result_dict
    text_annotation = ocr_text['result']['textAnnotation']
    full_text = text_annotation['fullText'] if 'fullText' in text_annotation else ""
    # нет текста - выходим
    if full_text == "":
        full_text = "There is no conversation on image"
        return full_text, result_dict
    image_width = int(text_annotation["width"])
    image_height = int(text_annotation["height"])
    result_dict["blocks_overall"] = len(text_annotation['blocks'])
    # храним данные предыдущего сообщения
    current_sentence = {
        "text": "",
        "coords": [0, 0, 0, 0],  # x1, y1, x2, y2
        "side": "middle",  # "response", "user" or "middle"
    }
    for block in text_annotation['blocks']:
        # если в блоке одна строка со временем, считаем её за временную отметку сообщения (надо?)
        is_time_block = len(block['lines']) == 1 and timestamp_pattern.match(block['lines'][0]['text']) is not None
        # смотрим насколько далеко был предыдущий блок (сравниваем нижнюю y-координату предыдущего
        # блока с верхней y-координатой текущего)
        # если блок расположен слишком близко, он может относиться
        # к предыдущей строке (или быть временной отметкой)
        current_y = int(block['boundingBox']['vertices'][0]['y'])  # текущая верхняя y-координата
        previous_block_too_close = current_y < current_sentence["coords"][3] + image_height * 0.02
        # если сообщение со временем и слишком близко - скорее всего это время сообщения, игнорируем
        if is_time_block and previous_block_too_close:
            continue
        # если левая координата блока находится в левой части картинки, то предполагаем что это ответ на сообщение
        if int(block['boundingBox']['vertices'][0]['x']) < image_width * block_percentile:
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
                    if (current_sentence['side'] != "middle" and
                        not (current_sentence["coords"][2] >= image_width * (1 - block_percentile / reduce_factor) and
                             current_sentence["coords"][0] <= image_width * (block_percentile / reduce_factor))):
                        result_dict["sentences"].append(current_sentence.copy())
                    current_sentence["text"] = ""
                    current_sentence["coords"] = get_coords_yandex(block)
                    current_sentence['side'] = "response"
            for line in block["lines"]:
                current_sentence["text"] += f"{line['text']} "
        # или если верхний правый угол блока находится справа, то предполагаем что это сообщение пользователя
        elif int(block['boundingBox']['vertices'][3]['x']) > image_width * (1 - block_percentile):
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
                    if (current_sentence["coords"][2] >= image_width * (1 - block_percentile) and
                            current_sentence["coords"][0] >= image_width * (block_percentile / reduce_factor)):
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
            else:
                # если предыдущий блок не центральный, а относится к какой-то части, то добавляем его
                if (current_sentence['side'] != "middle" and
                        not (current_sentence["coords"][2] >= image_width * (1 - block_percentile / reduce_factor) and
                             current_sentence["coords"][0] <= image_width * (block_percentile / reduce_factor))):
                    result_dict["sentences"].append(current_sentence.copy())
                current_sentence["text"] = ""
                current_sentence["coords"] = get_coords_yandex(block)
                current_sentence['side'] = "middle"
            for line in block["lines"]:
                current_sentence["text"] += f"{line['text']} "
    if (current_sentence['side'] != "middle" and
            not (current_sentence["coords"][2] >= image_width * (1 - block_percentile / reduce_factor) and
                 current_sentence["coords"][0] <= image_width * (block_percentile / reduce_factor))):
        result_dict["sentences"].append(current_sentence)
    return full_text, result_dict


def process_dict_yandex(ocr_dict: Dict, bounding_boxes: List) -> int:
    confidence = []
    for i, sent in enumerate(ocr_dict["sentences"]):
        temp_conf = 0
        for j, box in enumerate(bounding_boxes):
            if (sent['coords'][0] > box[0] and sent['coords'][2] < box[2] and sent['coords'][1] > box[1]
                    and sent['coords'][3] < box[3]):
                temp_conf = 1
                break

        if temp_conf == 0 and sent['side'] != 'middle':
            temp_conf = 0.20
        if sent['side'] != 'middle':
            confidence.append(temp_conf)

    if len(confidence) == 0:
        return 0
    return round(np.mean(confidence), 2)
