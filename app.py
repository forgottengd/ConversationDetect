import streamlit as st
from openai import OpenAI
import base64


def main():
    uploader = st.file_uploader("Choose image for detection", type=['png', 'jpg', 'jpeg'])
    if uploader:
        detect_button = st.empty()
        if detect_button.button("Detect"):
            detect_button.empty()
            on_click(uploader)


def on_click(file):
    client = OpenAI()
    base64_image = base64.b64encode(file.getvalue()).decode('utf-8')
    print(base64_image)
    with st.spinner("Detecting image..."):
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": "Analyze image. If it's conversation, try to understand conversation and describe it, otherwise shortly tell what you see on image"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

    st.text_area(label="Result", value=response.choices[0], height=300)


if __name__ == "__main__":
    main()
