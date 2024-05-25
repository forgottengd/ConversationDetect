from io import BytesIO
from typing import Tuple
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt, patches
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.measure import label, regionprops
import scipy.ndimage as nd


def resize_image(image, max_size=300):
    # Получаем текущие размеры изображения
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image
    # Вычисляем коэффициент масштабирования
    if height > width:
        scaling_factor = max_size / float(height)
    else:
        scaling_factor = max_size / float(width)

    # Новые размеры изображения
    new_size = (int(width * scaling_factor), int(height * scaling_factor))

    # Изменяем размер изображения с использованием Image.LANCZOS
    resized = image.resize(new_size, Image.LANCZOS)
    return resized


def get_bounding_boxes(file_path, block_percentile=0.18, reduce_factor=1.7):
    image = Image.open(file_path)
    # Приводим изображение в формат RGB
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    if image.mode == 'P':
        image = image.convert('RGBA').convert('RGB')
    image = resize_image(image, max_size=1300)
    gray_image = rgb2gray(image)
    # noise removal
    kernel = np.ones((2, 2), np.uint8)
    gray_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    gray_image = cv2.morphologyEx(gray_image, cv2.MORPH_DILATE, kernel, iterations=1)
    # Если изображение слишком светлое, повышаем контрастность изображения
    if np.mean(gray_image) > 0.96:
        sigma = 0.45
    elif np.mean(gray_image) > 0.93:
        sigma = 0.8
    else:
        sigma = 1.1
    # Выделяем края
    edges = canny(gray_image, sigma=sigma, low_threshold=0, high_threshold=0.2)
    # Заполняем дыры - для уменьшения шума при нахождении границ
    fill_im = nd.binary_fill_holes(edges)
    # Маркировка компонентов
    labeled_image, num_labels = label(fill_im, return_num=True)
    if num_labels > 1499:
        return [], None
    # Получение свойств каждой маркированной области
    props = regionprops(labeled_image)
    bounding_boxes = []
    min_box_width = image.width * 0.10
    min_box_height = image.height * 0.039
    for prop in props:
        y, x, max_row, max_col = prop.bbox
        # Рассчитываем width и height
        width = max_col - x
        height = max_row - y
        # не пропускаем мелкие блоки, а также смотрим на координаты блока,
        # принадлежит ли блок левой или правой стороне (но не обеим сторонам сразу с понижающим фактором)
        is_proper_size = (width > min_box_width and height > min_box_height)
        if is_proper_size:
            pass
        is_left_side = x <= image.width * block_percentile and (
                    x + width < image.width * (1 - block_percentile / reduce_factor))
        is_right_side = x + width >= image.width * (1 - block_percentile) and (x > image.width * (block_percentile / reduce_factor))
        if is_proper_size and is_left_side:
            bounding_boxes.append(((x, y, width, height), "left"))  # добавляем в список
        elif is_proper_size and is_right_side:
            bounding_boxes.append(((x, y, width, height), "right"))  # добавляем в список

    return bounding_boxes, fill_im


def process_image(file_path, block_percentile=0.18, reduce_factor=1.7) -> Tuple:
    bounding_boxes, fill_im = get_bounding_boxes(file_path, block_percentile, reduce_factor)
    if len(bounding_boxes) == 0:
        return 0, [], None
    messages = len(bounding_boxes)
    side_switches = 0
    _, last_side = bounding_boxes[0]
    for box, side in bounding_boxes[1:]:
        if last_side != side:
            side_switches += 1
            last_side = side

    return min((side_switches > 0) * 0.3 + side_switches * 0.10 + messages * 0.1, 1.0), bounding_boxes, fill_im


def plot_results(image, bounding_boxes):
    dpi = 300  # Разрешение в точках на дюйм, можно адаптировать
    height, width = image.shape
    figsize = width / float(dpi), height / float(dpi)  # Размер фигуры в дюймах
    # Создание фигуры и осей
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(image)
    for box, side in bounding_boxes:
        x, y, width, height = box
        # Отрисовываем все найденные блоки
        rect = patches.Rectangle((x, y), width, height, linewidth=0.8, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Убираем оси и белые поля
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Сохраняем фигуру в объект BytesIO для последующего использования в Streamlit
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    return buf
