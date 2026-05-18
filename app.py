import os
import requests
import pyperclip
import gradio as gr
from dotenv import load_dotenv
from langdetect import detect as detect_lang


load_dotenv("HF_TOKEN.env")

HF_TOKEN = os.getenv("HF_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

BASE_URL = "https://router.huggingface.co/hf-inference/models/Helsinki-NLP"

TO_EN = {
    "ar": "opus-mt-ar-en",
    "fr": "opus-mt-fr-en",
    "es": "opus-mt-es-en",
    "de": "opus-mt-de-en",
    "it": "opus-mt-it-en",
    "pt": "opus-mt-pt-en",
    "ru": "opus-mt-ru-en",
    "zh": "opus-mt-zh-en",
    "ja": "opus-mt-ja-en",
    "tr": "opus-mt-tr-en",
    "nl": "opus-mt-nl-en",
}

FROM_EN = {
    "ar": "opus-mt-en-ar",
    "fr": "opus-mt-en-fr",
    "es": "opus-mt-en-es",
    "de": "opus-mt-en-de",
    "it": "opus-mt-en-it",
    "pt": "opus-mt-en-pt",
    "ru": "opus-mt-en-ru",
    "zh": "opus-mt-zh",
    "ja": "opus-mt-en-jap",
    "tr": "opus-mt-en-tr",
    "nl": "opus-mt-en-nl",
}

LANG_NAMES = {
    "ar": "Arabic", "en": "English", "fr": "French",
    "es": "Spanish", "de": "German", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "zh": "Chinese",
    "ja": "Japanese", "tr": "Turkish", "nl": "Dutch",
}

LANG_CHOICES = [
    ("Arabic",     "ar"),
    ("English",    "en"),
    ("French",     "fr"),
    ("Spanish",    "es"),
    ("German",     "de"),
    ("Italian",    "it"),
    ("Portuguese", "pt"),
    ("Russian",    "ru"),
    ("Chinese",    "zh"),
    ("Japanese",   "ja"),
    ("Turkish",    "tr"),
    ("Dutch",      "nl"),
]


def hf_translate(text, model_name):
    url = f"{BASE_URL}/{model_name}"
    res = requests.post(url, headers=HEADERS, json={"inputs": text})
    data = res.json()
    if isinstance(data, dict) and "error" in data:
        raise Exception(data["error"])
    return data[0]["translation_text"]


def make_detected_html(lang="—"):
    return f'<p style="margin:0; padding:4px 0; font-size:14px;"><b>Detected Language:</b> {lang}</p>'


def translate(text, target_lang):
    if not text.strip():
        return "", "", make_detected_html(), ""

    try:
        text = text.strip()
        text = text[0].upper() + text[1:]

        src_lang = detect_lang(text)
        if src_lang not in LANG_NAMES:
            src_lang = "en"

        detected_name = LANG_NAMES.get(src_lang, src_lang)

        if src_lang == target_lang:
            return text, text, make_detected_html(detected_name), f"✅ Text is already in {detected_name}"

        if target_lang == "en":
            model = TO_EN.get(src_lang)
            if not model:
                return "", "", make_detected_html(detected_name), f"❌ No model found to translate from {detected_name} to English"
            result = hf_translate(text, model)

        elif src_lang == "en":
            model = FROM_EN.get(target_lang)
            if not model:
                return "", "", make_detected_html(detected_name), "❌ No model found for this target language"
            result = hf_translate(text, model)

        else:
            model_to_en = TO_EN.get(src_lang)
            model_to_target = FROM_EN.get(target_lang)
            if not model_to_en or not model_to_target:
                return "", "", make_detected_html(detected_name), "❌ No model found for this language pair"
            en_text = hf_translate(text, model_to_en)
            result = hf_translate(en_text, model_to_target)

        target_name = LANG_NAMES.get(target_lang, target_lang)
        status = f"✅ Detected: {detected_name} → Translated to: {target_name}"
        return result, result, make_detected_html(detected_name), status

    except Exception as e:
        return "", "", make_detected_html(), f"❌ Error: {str(e)}"


def clear_all():
    return "", "", make_detected_html(), "", "Cleared"


def copy_translation(text):
    if not text:
        return "⚠️ Nothing to copy"
    pyperclip.copy(text)
    return "✅ Copied to clipboard!"


css = """
.textbox-equal textarea {
    height: 200px !important;
    min-height: 200px !important;
}
"""

with gr.Blocks(title="Smart Translator", css=css) as app:
    gr.Markdown("# 🌐 Smart Translator\nAuto-detects input language and translates to your chosen language")

    translation_state = gr.State("")

    with gr.Row(equal_height=True):
        with gr.Column():
            detected_lang_html = gr.HTML(make_detected_html())
            input_text = gr.Textbox(
                label="Input Text",
                placeholder="Type here...",
                lines=7,
                elem_classes="textbox-equal"
            )
        with gr.Column():
            target_lang = gr.Dropdown(
                choices=LANG_CHOICES,
                value="ar",
                label="Output Language"
            )
            output_text = gr.Textbox(
                label="Translation",
                placeholder="Translation will appear here...",
                lines=7,
                interactive=False,
                elem_classes="textbox-equal"
            )

    with gr.Row():
        translate_btn = gr.Button("Translate", variant="primary")

    with gr.Row():
        clear_btn = gr.Button("Clear", variant="secondary")
        copy_btn = gr.Button("Copy Translation", variant="secondary")

    status_box = gr.Textbox(label="Status", interactive=False)

    input_text.change(
        fn=translate,
        inputs=[input_text, target_lang],
        outputs=[output_text, translation_state, detected_lang_html, status_box]
    )

    target_lang.change(
        fn=translate,
        inputs=[input_text, target_lang],
        outputs=[output_text, translation_state, detected_lang_html, status_box]
    )

    translate_btn.click(
        fn=translate,
        inputs=[input_text, target_lang],
        outputs=[output_text, translation_state, detected_lang_html, status_box]
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[input_text, output_text, detected_lang_html, translation_state, status_box]
    )

    copy_btn.click(
        fn=copy_translation,
        inputs=[translation_state],
        outputs=[status_box]
    )

app.launch(theme=gr.themes.Soft())



#  cd "C:\2_ projects\NLP\Translate"
#  python -m gradio app_local.py

