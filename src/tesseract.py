import os
from typing import Dict, Tuple
import cv2
import numpy as np
import pytesseract


def parse_ocr_tesseract(processed_result) -> Tuple:
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


def process_image_tesseract(details, image_width, image_height, block_percentile) -> Dict:
    text_blocks = []  # Список для хранения информации о текстовых блоках
    reduce_factor = 1.37  # коэффициент для уменьшения границ для обнаружения широких блоков
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


def parse_image_tesseract(file, from_bytes=True):
    # устанавливаем путь до тессеракта для Windows
    if os.name == 'nt':
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if from_bytes:
        # Читаем файл как массив байт
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
    else:
        # Загрузка изображения по пути файла
        image = cv2.imread(file)

    # Перевод в серый для улучшения распознавания
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Используем Tesseract для распознавания текста и получения детальной информации о расположении текста
    details = pytesseract.image_to_data(gray_image, lang='eng+rus', config=r'--oem 3 --psm 4',
                                        output_type=pytesseract.Output.DICT)
    return details, image.shape[1], image.shape[0]
