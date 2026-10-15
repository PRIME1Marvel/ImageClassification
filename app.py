import streamlit as st
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import time

st.set_page_config(
    page_title="NeuralVision | Cat vs Dog",
    page_icon="🐾",
    layout="wide",
)

# ImageNet synsets used by MobileNetV2 for common domestic-cat classes.
CAT_SYNSETS = {
    "n02123045",  # tabby
    "n02123159",  # tiger cat
    "n02123394",  # Persian cat
    "n02123597",  # Siamese cat
    "n02124075",  # Egyptian cat
}

# ImageNet dog-breed synsets generally fall in this range.
# The model's decoded label is also checked for common dog-related words.
DOG_WORDS = {
    "dog", "retriever", "terrier", "shepherd", "spaniel", "poodle",
    "husky", "malamute", "hound", "chihuahua", "pug", "rottweiler",
    "doberman", "boxer", "beagle", "collie", "corgi", "dachshund",
    "pinscher", "schnauzer", "mastiff", "wolfhound", "setter", "pointer",
    "briard", "kelpie", "komondor", "kuvasz", "samoyed", "papillon",
    "pekinese", "peke", "basenji", "borzoi", "whippet", "greyhound",
    "affenpinscher", "mexican hairless", "shih-tzu", "shih tzu",
}

SAMPLES = {
    "Tabby cat": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?auto=format&fit=crop&w=900&q=85",
    "Golden retriever": "https://images.unsplash.com/photo-1552053831-71594a27632d?auto=format&fit=crop&w=900&q=85",
    "Cat portrait": "https://images.unsplash.com/photo-1513360371669-4adf3dd7dff8?auto=format&fit=crop&w=900&q=85",
    "German shepherd": "https://images.unsplash.com/photo-1589941013453-ec89f33b5e95?auto=format&fit=crop&w=900&q=85",
}

@st.cache_resource(show_spinner="Loading pretrained MobileNetV2 model…")
def load_model():
    # TensorFlow downloads ImageNet weights the first time the app runs.
    from tensorflow.keras.applications import MobileNetV2
    model = MobileNetV2(weights="imagenet")
    return model

def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

def predict_image(image: Image.Image):
    from tensorflow.keras.applications.mobilenet_v2 import (
        preprocess_input,
        decode_predictions,
    )

    model = load_model()
    resized = image.convert("RGB").resize((224, 224))
    array = np.asarray(resized, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)
    batch = preprocess_input(batch)
    probabilities = model.predict(batch, verbose=0)
    top = decode_predictions(probabilities, top=5)[0]

    # top item: (synset ID, human-readable label, probability)
    top_synset, top_label, top_probability = top[0]
    normalized_label = top_label.lower().replace("_", " ")

    if top_synset in CAT_SYNSETS or "cat" in normalized_label:
        predicted_class = "Cat"
    elif any(word in normalized_label for word in DOG_WORDS):
        predicted_class = "Dog"
    elif top_synset.startswith("n020") or top_synset.startswith("n021"):
        # Many ImageNet dog breeds have breed-only names; the range covers
        # the standard ImageNet dog-breed synsets. The label is shown too.
        try:
            synset_num = int(top_synset[1:])
        except ValueError:
            synset_num = 0
        if 2085620 <= synset_num <= 2116738:
            predicted_class = "Dog"
        else:
            predicted_class = "Other"
    else:
        predicted_class = "Other"

    return predicted_class, float(top_probability), top

def get_image_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
      .hero {padding: 1.2rem 1.4rem; border-radius: 18px;
             background: linear-gradient(120deg, #312e81 0%, #6d28d9 55%, #0f766e 100%);
             color: white; margin-bottom: 1rem;}
      .hero h1 {margin: 0; padding: 0; font-size: 2rem;}
      .hero p {margin: .35rem 0 0 0; opacity: .88;}
      .metric-note {font-size: .85rem; color: #9ca3af;}
    </style>
    <div class="hero">
      <h1>🐾 NeuralVision</h1>
      <p>Cat vs Dog image analysis · MobileNetV2 pretrained on ImageNet</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "This app uses a real pretrained ImageNet classifier. It is not your separately "
    "trained Cat-vs-Dog model. The displayed confidence is the top ImageNet class "
    "probability, not a calibrated Cat-vs-Dog probability."
)

with st.sidebar:
    st.header("Image source")
    source = st.radio("Choose input", ["Upload image", "Image URL", "Sample gallery"])
    st.divider()
    st.caption("Accepted formats: JPG, JPEG, PNG, WEBP")

image = None
if source == "Upload image":
    uploaded = st.file_uploader("Upload a cat or dog image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
        except Exception as exc:
            st.error(f"Could not read this image: {exc}")
elif source == "Image URL":
    url = st.text_input("Paste a direct image URL", placeholder="https://example.com/image.jpg")
    if st.button("Load image", type="primary"):
        if not url.strip().lower().startswith(("http://", "https://")):
            st.error("Please enter a valid http:// or https:// URL.")
        else:
            try:
                image = load_image_from_url(url.strip())
                st.session_state["url_image"] = get_image_bytes(image)
            except Exception as exc:
                st.error(f"Could not load image: {exc}")
    if image is None and st.session_state.get("url_image"):
        image = Image.open(BytesIO(st.session_state["url_image"])).convert("RGB")
else:
    choice = st.selectbox("Choose a sample image", list(SAMPLES.keys()))
    if st.button("Use sample image", type="primary"):
        try:
            image = load_image_from_url(SAMPLES[choice])
            st.session_state["sample_image"] = get_image_bytes(image)
        except Exception as exc:
            st.error(f"Could not load sample image: {exc}")
    if image is None and st.session_state.get("sample_image"):
        image = Image.open(BytesIO(st.session_state["sample_image"])).convert("RGB")

if image is not None:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Selected image")
        st.image(image, use_container_width=True)
        st.download_button(
            "Download selected image",
            data=get_image_bytes(image),
            file_name="selected_image.png",
            mime="image/png",
        )
    with right:
        st.subheader("Prediction")
        if st.button("Classify image", type="primary", use_container_width=True):
            try:
                with st.spinner("Running MobileNetV2 inference…"):
                    start = time.perf_counter()
                    label, confidence, top = predict_image(image)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                st.session_state["last_prediction"] = {
                    "label": label,
                    "confidence": confidence,
                    "elapsed_ms": elapsed_ms,
                    "top": [(name.replace("_", " "), float(score)) for _, name, score in top],
                }
            except Exception as exc:
                st.error(
                    "Prediction failed. Check that TensorFlow is installed and that "
                    f"the model weights can be downloaded. Details: {exc}"
                )

        result = st.session_state.get("last_prediction")
        if result:
            label = result["label"]
            if label == "Cat":
                st.success("🐱 Prediction: Cat")
            elif label == "Dog":
                st.success("🐶 Prediction: Dog")
            else:
                st.warning("Prediction: Other / uncertain")
            st.metric("Top-class confidence", f"{result['confidence'] * 100:.2f}%")
            st.caption(f"Inference time: {result['elapsed_ms']:.0f} ms (includes any first-load overhead)")
            st.markdown("**Top ImageNet predictions**")
            for name, score in result["top"]:
                st.write(f"{name}: {score * 100:.2f}%")
                st.progress(min(max(score, 0.0), 1.0))
else:
    st.markdown("### Try it out")
    st.write("Choose an image source from the sidebar, then upload or load an image.")
    st.caption("No predictions are made until you select an image and click Classify image.")

with st.expander("About this demo"):
    st.write(
        "MobileNetV2 is a pretrained image-classification model. This app maps selected "
        "ImageNet labels to Cat or Dog and shows the model's top five ImageNet classes. "
        "It is a working demo, but it is not a replacement for training and validating "
        "a dedicated binary classifier on your own cat-and-dog dataset."
    )
