import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from diffusers import StableDiffusionPipeline

# Initialize translation model
translation_model_name = "utrobinmv/t5_translate_en_ru_zh_large_1024"
translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name)

# Initialize image generation model
image_model_name = "CompVis/stable-diffusion-v1-4"
image_pipeline = StableDiffusionPipeline.from_pretrained(image_model_name)

# Move models to CPU
translation_model = translation_model.to("cpu")
image_pipeline = image_pipeline.to("cpu")

# Default prompt
default_prompt = "Генерация дизайна логотипа студенческого СМИ"

st.title("Логотип Генератор")

# Input for prompt
prompt_ru = st.text_area("Введите текст на русском языке:", value=default_prompt)

def translate_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {key: value.to("cpu") for key, value in inputs.items()}  # Ensure inputs are on CPU
    outputs = model.generate(inputs["input_ids"], max_length=1024)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

def generate_image(prompt, pipeline):
    image = pipeline(prompt).images[0]
    return image

if st.button("Получить логотип"):
    with st.spinner('Перевод текста...'):
        prompt_en = translate_text(prompt_ru, translation_model, translation_tokenizer)
        st.write(f"Переведенный текст: {prompt_en}")

    with st.spinner('Генерация изображения...'):
        image = generate_image(prompt_en, image_pipeline)
        st.image(image, caption="Сгенерированный логотип")

# To run this script, save it as app.py and run the following command:
# streamlit run app.py
