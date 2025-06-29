import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
from core.model import TrainedModel

st.set_page_config(page_title="NeuroVision Digit Classifier", layout="centered")
st.title("Draw a Digit - NeuroVision Classifier")

# --- Sidebar settings ---
st.sidebar.header("Canvas Settings")
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 15)
invert_input = st.sidebar.checkbox("Invert Input (if prediction is bad)", value=False)

# --- Drawing canvas ---
st.write("Draw a digit (0-9) below:")
canvas_result = st_canvas(
    fill_color="#000000",      # Ink color when filling (black strokes)
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",    # White pen
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# --- Prediction ---
if canvas_result.image_data is not None:
    image = canvas_result.image_data

    # Convert to PIL grayscale image
    image_pil = Image.fromarray((image[:, :, 0]).astype(np.uint8))
    image_pil = image_pil.resize((28, 28)).convert('L')  # MNIST style

    if invert_input:
        image_pil = ImageOps.invert(image_pil)  # Optional inversion toggle

    # Normalize
    image_array = np.array(image_pil).astype(np.float32) / 255.0
    image_flattened = image_array.flatten()

    # Display processed image
    st.image(image_pil, width=100, caption="Processed Input (28×28)")

    # Model inference
    model = TrainedModel()
    prediction, confidences = model.predict(image_flattened, return_confidence=True)

    st.success(f"Predicted Digit: {prediction}")
    
    st.subheader("Confidence Scores:")
    for i, conf in enumerate(confidences):
        st.write(f"Digit {i}: {conf:.4f}")
