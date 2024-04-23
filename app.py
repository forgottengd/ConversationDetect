import streamlit as st
from transformers import pipeline
from PIL import Image


def main():
    uploader = st.file_uploader("Choose image for detection", type=['png', 'jpg', 'jpeg', 'webp'])
    if uploader:
        detect_button = st.empty()
        if detect_button.button("Detect"):
            detect_button.empty()
            on_click(uploader)


def on_click(file):
    image = Image.open(file)
    with st.spinner("Detecting image..."):
        model_id = "llava-hf/llava-1.5-7b-hf"
        pipe = pipeline("image-to-text", model=model_id)
        prompt = "USER: <image>\nDescribe image. If it's conversation, try to understand conversation context\nASSISTANT:"
        outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 250})

    st.text_area(label="Result", value=outputs[0]['generated_text'][96:], height=300)


if __name__ == "__main__":
    main()
