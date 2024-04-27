import base64
import os
from typing import Dict
import requests
import streamlit as st
import re
#import res


OCR_API = os.environ['OCR_API']
#OCR_API = ""
folder_id = "b1g79vtsge6ggong8ioa"
block_percentile = 0.19  # зона в которую должен входить блок, чтобы считать его за переписку


def encode_file_to_base64(file):
    """ Кодирует файл изображения в строку base64. """
    encoded_content = base64.b64encode(file)
    return encoded_content.decode('utf-8')


def send_ocr_request(iam_token, encoded_image):
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


def process_ocr(ocr_text, temp):
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
                "x1": 0, "x2": 0, "y1": 0, "y2": 0,
                "side": ""  # "response" or "user"
            }
            for block in text_annotation['blocks']:
                # если в блоке одна строка со временем, считаем её за временную отметку сообщения
                is_time_block = len(block['lines']) == 1 and timestamp_pattern.match(block['lines'][0]['text']) is not None
                # смотрим насколько далеко был предыдущий блок (сравниваем нижнюю y-координату предыдущего
                # блока с верхней y-координатой текущего)
                # если блок расположен слишком близко, он может относиться
                # к предыдущей строке (или быть временной отметкой)
                current_y = int(block['boundingBox']['vertices'][0]['y'])  # текущая верхняя y-координата
                previous_block_too_close = current_y < current_sentence["y2"] + result_dict["image_height"] * 0.02
                # если верхний левый угол блока находится слева, то предполагаем что это ответ на сообщение
                if int(block['boundingBox']['vertices'][0]['x']) < result_dict["image_width"] * block_percentile:
                    # если текущее предложение отсутствует
                    if current_sentence["text"] == "":
                        current_sentence["x1"] = int(block['boundingBox']['vertices'][0]['x'])
                        current_sentence["y1"] = int(block['boundingBox']['vertices'][0]['y'])
                        current_sentence["x2"] = int(block['boundingBox']['vertices'][2]['x'])
                        current_sentence["y2"] = int(block['boundingBox']['vertices'][2]['y'])
                        current_sentence['side'] = "response"
                    else:
                        # если блок расположен близко к предыдущему, считаем что он ему принадлежит
                        if previous_block_too_close:
                            current_sentence["x2"] = max(int(block['boundingBox']['vertices'][2]['x']), current_sentence["x2"])
                            current_sentence["y2"] = max(int(block['boundingBox']['vertices'][2]['y']), current_sentence["y2"])
                        # если это новый блок, добавляем предыдущее сообщение и создаем новое
                        else:
                            result_dict["sentences"].append(current_sentence.copy())
                            current_sentence["text"] = ""
                            current_sentence["x1"] = int(block['boundingBox']['vertices'][0]['x'])
                            current_sentence["y1"] = int(block['boundingBox']['vertices'][0]['y'])
                            current_sentence["x2"] = int(block['boundingBox']['vertices'][2]['x'])
                            current_sentence["y2"] = int(block['boundingBox']['vertices'][2]['y'])
                            current_sentence['side'] = "response"
                    for line in block["lines"]:
                        current_sentence["text"] += f"{line['text']} "
                # или если верхний правый угол блока находится справа, то предполагаем что это сообщение пользователя
                elif int(block['boundingBox']['vertices'][3]['x']) > result_dict["image_width"] * (1 - block_percentile):
                    # если текущее предложение отсутствует
                    if current_sentence["text"] == "":
                        current_sentence["x1"] = int(block['boundingBox']['vertices'][0]['x'])
                        current_sentence["y1"] = int(block['boundingBox']['vertices'][0]['y'])
                        current_sentence["x2"] = int(block['boundingBox']['vertices'][2]['x'])
                        current_sentence["y2"] = int(block['boundingBox']['vertices'][2]['y'])
                        current_sentence['side'] = "user"
                    else:
                        # если блок расположен близко к предыдущему, считаем что он ему принадлежит
                        if previous_block_too_close:
                            # расширяем х- и y-координаты
                            current_sentence["x1"] = max(int(block['boundingBox']['vertices'][0]['x']), current_sentence["x1"])
                            current_sentence["x2"] = max(int(block['boundingBox']['vertices'][2]['x']), current_sentence["x2"])
                            current_sentence["y2"] = max(int(block['boundingBox']['vertices'][2]['y']), current_sentence["y2"])
                            # проверяем на сторону, возможно предыдущий блок был слишком близко расположен к процентилю
                            # границы и неправильно записан в response. проверяем по block_percentile / 2
                            # если ближе к правой границе чем к левой то меняем сторону
                            if (current_sentence["x2"] >= result_dict["image_width"] * (1 - block_percentile / 2) and
                                    current_sentence["x1"] >= result_dict["image_width"] * (block_percentile / 2)):
                                current_sentence['side'] = "user"
                        else:
                            result_dict["sentences"].append(current_sentence.copy())
                            current_sentence["text"] = ""
                            current_sentence["x1"] = int(block['boundingBox']['vertices'][0]['x'])
                            current_sentence["y1"] = int(block['boundingBox']['vertices'][0]['y'])
                            current_sentence["x2"] = int(block['boundingBox']['vertices'][2]['x'])
                            current_sentence["y2"] = int(block['boundingBox']['vertices'][2]['y'])
                            current_sentence['side'] = "user"
                    for line in block["lines"]:
                        current_sentence["text"] += f"{line['text']} "
                # если блок не входит в процентную зону краев картинки
                else:
                    # TODO: возможно если блок в центре и не относится к какой-то стороне, его можно соотнести потом -
                    #  проверить. Также если блок в центре, и не находит подтверждение к сторонам, статус блока? добавлять?
                    # если блок расположен близко к предыдущему, считаем что он ему принадлежит
                    if previous_block_too_close:
                        # расширяем границы текущего блока
                        current_sentence["x1"] = max(int(block['boundingBox']['vertices'][0]['x']), current_sentence["x1"])
                        current_sentence["x2"] = max(int(block['boundingBox']['vertices'][2]['x']), current_sentence["x2"])
                        current_sentence["y2"] = max(int(block['boundingBox']['vertices'][2]['y']), current_sentence["y2"])
                        for line in block["lines"]:
                            current_sentence["text"] += f"{line['text']} "
            result_dict["sentences"].append(current_sentence)
    return full_text, result_dict


def process_dict(ocr_dict: Dict):
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


def main():
    """ Создает объекты streamlit-сервиса """
    uploader = st.file_uploader("Choose image for detection", type=['png', 'jpg', 'jpeg', 'webp'])
    if uploader:
        slider_value = st.slider(
            'Threshold',
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            value=0.8
        )
        detect_button = st.empty()
        if detect_button.button("Detect"):
            detect_button.empty()
            on_click(uploader, slider_value)


def on_click(file, temp):
    """ Вызывает обработку загруженного изображения и отображает результат """
    with st.spinner("Detecting image..."):
        try:
            image = encode_file_to_base64(file.getvalue())
            ocr_response = send_ocr_request(OCR_API, image)
            # parse OCR result
            #ocr_response = res.res4
            full_text, result_dict = process_ocr(ocr_response, temp)
            metrics = process_dict(result_dict)
            confidence = (metrics[0] * 0.15 + metrics[1] * 0.03 + metrics[2] * 0.03) * 2
            decision = "Переписка" if confidence >= temp else "Не переписка"
            decision += f". Уверенность: {confidence}"
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.text_area(label="Result", value=decision, height=200)


if __name__ == "__main__":
    #print(on_click("", 0.8))
    main()
