import base64
import requests
import streamlit as st
from openai import OpenAI


client = OpenAI()


def encode_file_to_base64(file):
    """ Кодирует файл изображения в строку base64. """
    encoded_content = base64.b64encode(file)
    return encoded_content.decode('utf-8')


def send_ocr_request(iam_token, encoded_image):
    """ Отправляет OCR запрос и возвращает результат в json формате. """
    folder_id = "b1g79vtsge6ggong8ioa"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {iam_token}",
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


def main():
    uploader = st.file_uploader("Choose image for detection", type=['png', 'jpg', 'jpeg'])
    if uploader:
        slider_value = st.slider(
            'Temperature',
            min_value=0.0,  # минимальное значение
            max_value=2.0,  # максимальное значение
            step=0.1,  # шаг изменения
            value=1.0  # начальное значение
        )
        detect_button = st.empty()
        if detect_button.button("Detect"):
            detect_button.empty()
            on_click(uploader, slider_value)


def summary_prompt(ocr_text, blocks) -> str:
    text = ("There is text, which I got by optical character recognition. You should do deep analyse "
            "and understand is it converstaion or not. Remember, you should answer with one "
            f"and only one word: yes or no. Text to analyse: {ocr_text}")
    return text


def on_click(file, temp):
    IAM_TOKEN = 't1.9euelZqMnoqJzo6Jk8qbk5GXmsuOy-3rnpWaz5CUmJaelpKUk8nHjY2Ujonl8_cIL1xO-e9laE83_N3z90hdWU7572VoTzf8zef1656VmpeLms-Ri5OexpvOjsyZjYqU7_zF656VmpeLms-Ri5OexpvOjsyZjYqU.OeOAbd345tFFCLOSg2uj7hgBbspIHLR-j8BVVv5JfvnBtWWgkd1bzBe-GkJACxzzB2fzP_dJ7Qa1jW8vzaTiBA'

    full_text = ""
    with st.spinner("Detecting image..."):
        try:
            image = encode_file_to_base64(file.getvalue())
            ocr_response = send_ocr_request(IAM_TOKEN, image)
            # parse OCR result
            if 'result' in ocr_response:
                if 'textAnnotation' in ocr_response['result']:
                    text_annotation = ocr_response['result']['textAnnotation']
                    full_text = text_annotation['fullText'] if 'fullText' in text_annotation else ""
                    blocks_number = len(text_annotation['blocks'])

            if full_text == "":
                full_text = "There is no conversation on image"
            else:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "user",
                            "content": summary_prompt(full_text, blocks_number),
                        }
                    ],
                    max_tokens=300,
                    temperature=temp,
                )
                full_text = response.choices[0].message["content"]
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # st.text_area(label="Result", value=response.choices[0].message["content"], height=300)
    st.text_area(label="Result", value=full_text, height=300)


if __name__ == "__main__":
    main()
