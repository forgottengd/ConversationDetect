import os
import pytest
from src.image_processing import process_image


def get_image_files(directory):
    """
    Возвращает список путей к изображениям jpg и png в указанной директории.
    """
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg')):
                image_files.append(os.path.join(root, file))
    return image_files


@pytest.mark.parametrize("image_file", get_image_files("imgs"))
def test_image_processing(image_file):
    """
    Тест проверяет, что функция process_image успешно обрабатывает изображение без поднятия исключений.
    pytest.mark.parametrize используется для того, чтобы создать отдельный тест для каждого файла.
    """
    confidence_level = 0.7
    confidence, _, _ = process_image(image_file)
    if "imgs\\0\\" in image_file:
        assert confidence < confidence_level, f"Confidence TOO HIGH for {image_file} with confidence {confidence}"
    else:
        assert confidence >= confidence_level, f"Processing failed for {image_file} with confidence {confidence}"
