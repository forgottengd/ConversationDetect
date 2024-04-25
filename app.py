import base64
import os
import requests
import streamlit as st


OCR_API = os.environ['OCR_API']
folder_id = "b1g79vtsge6ggong8ioa"


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


def send_gpt_request(ocr_token, ocr_text, temp):
    """ Отправляет GPT запрос и возвращает результат в json формате. """
    print(ocr_text)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {ocr_token}",
        "x-folder-id": folder_id,
    }

    body = {
        "modelUri": f"gpt://{folder_id}/yandexgpt/latest",
        "completionOptions": {
            "stream": False,
            "temperature": temp,
            "maxTokens": "100"
        },
        "messages": [
            {
                "role": "system",
                "text": "Проведи глубокий анализ текста и внимательно посмотри, является ли он перепиской. "
                        "Используй эти признаки переписки: текст выглядит как диалог или общение, "
                        "в нем есть временные метки сообщений, приветствия, прощания, или какое-то общение."
                        "Ответь одним и только одним словом: да или нет"
            },
            {
                "role": "user",
                "text": ocr_text
            }
        ]
    }

    response = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/completion", headers=headers,
                             json=body)
    return response.json()


def main():
    """ Создает объекты streamlit-сервиса """
    uploader = st.file_uploader("Choose image for detection", type=['png', 'jpg', 'jpeg'])
    if uploader:
        slider_value = st.slider(
            'Temperature',
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            value=0.4
        )
        detect_button = st.empty()
        if detect_button.button("Detect"):
            detect_button.empty()
            on_click(uploader, slider_value)


def on_click(file, temp):
    """ Вызывает обработку загруженного изображения и отображает результат """
    full_text = ""
    with st.spinner("Detecting image..."):
        try:
            image = encode_file_to_base64(file.getvalue())
            ocr_response = send_ocr_request(OCR_API, image)
            # parse OCR result
            if 'result' in ocr_response:
                if 'textAnnotation' in ocr_response['result']:
                    text_annotation = ocr_response['result']['textAnnotation']
                    full_text = text_annotation['fullText'] if 'fullText' in text_annotation else ""
                    blocks_number = len(text_annotation['blocks'])

            if full_text == "":
                full_text = "There is no conversation on image"
            else:
                gpt_response = send_gpt_request(OCR_API, full_text, temp)
                full_text = gpt_response['result']['alternatives'][0]['message']['text']
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.text_area(label="Result", value=full_text, height=300)


if __name__ == "__main__":
    main()
