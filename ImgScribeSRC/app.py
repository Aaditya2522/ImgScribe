import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch


# Load model and processor
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("./blip-caption-model")
    model = BlipForConditionalGeneration.from_pretrained("./blip-caption-model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device


processor, model, device = load_model()

# Set dark mode page config
st.set_page_config(page_title="🧠 ImgScribe", layout="centered", page_icon="🖼️")

# Custom dark theme styling
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0e1117;
            color: #e1e1e1;
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            font-size: 2.8em;
            font-weight: 700;
            color: #f7f7f7;
            text-align: center;
            margin-top: 30px;
        }
        .subtitle {
            font-size: 1.1em;
            text-align: center;
            color: #b0b0b0;
            margin-bottom: 30px;
        }
        .caption-box {
            background-color: #1e1f26;
            border-radius: 10px;
            padding: 15px;
            margin-top: 25px;
            text-align: center;
            font-size: 1.2em;
            color: #ffffff;
            box-shadow: 0 2px 12px rgba(255,255,255,0.05);
        }
        .footer {
            font-size: 0.9em;
            text-align: center;
            color: #6e6e6e;
            margin-top: 50px;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# Title
st.markdown(
    '<div class="title">🧠 ImgScribe: AI-Powered Image Describer</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Upload an image and let BLIP generate a meaningful caption.</div>',
    unsafe_allow_html=True,
)

# Upload section
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(image, caption="Preview", use_container_width=True)

    if st.button("📝 Generate Caption"):
        with st.spinner("Generating caption..."):
            inputs = processor(images=image, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=30)
            caption = processor.decode(out[0], skip_special_tokens=True)

        st.markdown(f'<div class="caption-box">{caption}</div>', unsafe_allow_html=True)

# Footer
st.markdown(
    '<div class="footer">⚡ Built with BLIP + Streamlit • © 2025 CaptionCraft</div>',
    unsafe_allow_html=True,
)
