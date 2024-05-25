import base64
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from typing import Tuple
from io import BytesIO
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.measure import label, regionprops
import scipy.ndimage as nd


def encode_file_to_base64(file) -> str:
    """ Кодирует файл изображения в строку base64. """
    encoded_content = base64.b64encode(file)
    return encoded_content.decode('utf-8')


def elapsed_time(start_time, end_time) -> str:
    """ Получает разницу во времени и возвращает форматированное значение """
    elapsed = end_time - start_time
    if elapsed < 1:
        return f"{elapsed * 1000:.2f} мс"
    return f"{elapsed:.2f} секунд"


def draw_rectangles_yandex(image_path, blocks, sentences) -> BytesIO:
    """
    Отображает прямоугольники на изображении.

    :param sentences: список найденных предложений с прямоугольниками (x1, x2, y1, y2)
    :param image_path: путь к файлу изображения.
    :param blocks: список прямоугольников, где каждый прямоугольник задается как (x, y, width, height).
    """
    # Загрузка изображения
    img = Image.open(image_path)
    if img.mode == 'P':
        img = img.convert('RGBA').convert('RGB')
    img_np = np.array(img)

    dpi = 300  # Разрешение в точках на дюйм, можно адаптировать
    height, width, _ = img_np.shape
    figsize = width / float(dpi), height / float(dpi)  # Размер фигуры в дюймах
    # Создание фигуры и осей
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(img_np)

    # Добавление прямоугольников OCR
    for block in blocks:
        x = int(block['boundingBox']['vertices'][0]['x'])
        y = int(block['boundingBox']['vertices'][0]['y'])
        width = int(block['boundingBox']['vertices'][2]['x']) - x
        height = int(block['boundingBox']['vertices'][2]['y']) - y
        rect = patches.Rectangle((x, y), width, height, linewidth=0.9, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Добавление прямоугольников Processed_OCR
    for sent in sentences:
        rect = patches.Rectangle((sent["coords"][0], sent["coords"][1]),
                                 sent["coords"][2] - sent["coords"][0],
                                 sent["coords"][3] - sent["coords"][1],
                                 linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    # Убираем оси и белые поля
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Сохраняем фигуру в объект BytesIO для последующего использования в Streamlit
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    # Возвращаем объект BytesIO, который можно отобразить в Streamlit
    return buf


def draw_rectangles_tesseract(image_path, ocr_result, text_blocks) -> BytesIO:
    """
    Отображает прямоугольники на изображении.

    :param image_path: путь к файлу изображения.
    :param ocr_result: результат обработки OCR
    :param text_blocks: список обработанных предложений, где каждый прямоугольник задается как (x1, x2, y1, y2).
    """
    # Загрузка изображения
    img = Image.open(image_path)
    img_np = np.array(img)

    dpi = 300  # Разрешение в точках на дюйм, можно адаптировать
    height, width, _ = img_np.shape
    figsize = width / float(dpi), height / float(dpi)  # Размер фигуры в дюймах
    # Создание фигуры и осей
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(img_np)

    # Добавление сырых прямоугольников OCR
    for i in range(len(ocr_result['text'])):
        if ocr_result['text'][i] != '':
            rect = patches.Rectangle((ocr_result["left"][i], ocr_result["top"][i]),
                                     ocr_result["width"][i], ocr_result["height"][i],
                                     linewidth=0.7, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    # Добавление обработанных прямоугольников OCR
    for sent in text_blocks:
        rect = patches.Rectangle((sent["coords"][0], sent["coords"][1]),
                                 sent["coords"][2] - sent["coords"][0],
                                 sent["coords"][3] - sent["coords"][1],
                                 linewidth=0.9, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    # Убираем оси и белые поля
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Сохраняем фигуру в объект BytesIO для последующего использования в Streamlit
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    # Возвращаем объект BytesIO, который можно отобразить в Streamlit
    return buf


def get_color_zones(file, ocr_responce, block_percentile=0.19, reduce_factor=1.7) -> Tuple:
    image = Image.open(file)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    if image.mode == 'P':
        image = image.convert('RGBA').convert('RGB')
    gray_image = rgb2gray(image)
    # noise removal
    kernel = np.ones((1, 1), np.uint8)
    gray_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Если изображение слишком светлое, повышаем контрастность изображения
    if np.mean(gray_image) > 0.93:
        sigma = 0.8
    else:
        sigma = 1.1
    # Выделяем края
    edges = canny(gray_image, sigma=sigma, low_threshold=0)
    # Убираем из краев найденный текст - таким образом уменьшаем шум для заполнения краев
    for block in ocr_responce['result']['textAnnotation']['blocks']:
        x, y, width, height = (int(block['boundingBox']['vertices'][0]['x']),
                               int(block['boundingBox']['vertices'][0]['y']),
                               int(block['boundingBox']['vertices'][2]['x']) - int(
                                   block['boundingBox']['vertices'][0]['x']),
                               int(block['boundingBox']['vertices'][2]['y']) - int(
                                   block['boundingBox']['vertices'][0]['y']))
        edges[y:y + height, x:x + width] = False
    # Заполняем дыры - для уменьшения шума при нахождении границ
    fill_im = nd.binary_fill_holes(edges)
    # Маркировка компонентов
    labeled_image = label(fill_im)
    # Получение свойств каждой маркированной области
    props = regionprops(labeled_image)

    dpi = 300  # Разрешение в точках на дюйм, можно адаптировать
    height, width = image.height, image.width
    figsize = width / float(dpi), height / float(dpi)  # Размер фигуры в дюймах
    # Создание фигуры и осей
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(fill_im)
    bounding_boxes = []
    for prop in props:
        y, x, max_row, max_col = prop.bbox
        # Рассчитываем width и height
        width = max_col - x
        height = max_row - y
        # не пропускаем мелкие блоки, а также смотрим на координаты блока,
        # принадлежит ли блок левой или правой стороне (но не обеим сторонам сразу с понижающим фактором)
        if ((width > 17 and height > 17)
                and (x <= image.width * block_percentile or x + width >= image.width * (1 - block_percentile))
                and not (x <= image.width * (block_percentile / reduce_factor) and
                         x + width >= image.width * (1 - block_percentile / reduce_factor))):
            rect = patches.Rectangle((x, y),
                                     width, height,
                                     linewidth=0.8, edgecolor='r', facecolor='none')
            ax.add_patch(rect)  # добавляем на картинку
            bounding_boxes.append((x, y, x + width, y + height))  # добавляем в список

    # Убираем оси и белые поля
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf, bounding_boxes
