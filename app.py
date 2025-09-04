import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import io

# ----------------------------
# Streamlit Config
# ----------------------------
st.set_page_config(page_title="YOLOv11 Brain Tumor Detector", page_icon="🧠", layout="wide")
st.title("🧠 Brain Tumor Detection using YOLOv11")
st.write("Upload MRI scan images to detect **Pituitary, No Tumor, Meningioma, Glioma**.")

# ----------------------------
# Load YOLOv11 Model
# ----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Replace with your trained model path

model = load_model()

# ----------------------------
# Helper Functions
# ----------------------------
def preprocess_image(pil_img):
    img_np = np.array(pil_img)
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    return img_np

def get_detection_data(result):
    boxes = result.boxes
    data = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            data.append({
                "Class": model.names[cls_id],
                "Confidence": round(conf, 3),
                "Xmin": int(xyxy[0]),
                "Ymin": int(xyxy[1]),
                "Xmax": int(xyxy[2]),
                "Ymax": int(xyxy[3])
            })
    return data

# ----------------------------
# File Upload and Inference
# ----------------------------
uploaded_files = st.file_uploader(
    "📁 Choose one or more MRI images...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"📤 Uploaded: {uploaded_file.name}", use_column_width=True)

        with st.spinner("🔍 Detecting tumors..."):
            results = model(image)
        result = results[0]

        plotted_img = result.plot()

        st.subheader(f"🧠 Detection Result: {uploaded_file.name}")
        st.image(plotted_img, caption="Detected Image", use_column_width=True)

        # Convert image to bytes for download
        img_bytes = io.BytesIO()
        Image.fromarray(plotted_img).save(img_bytes, format="PNG")
        img_bytes.seek(0)

        st.download_button(
            label="💾 Download Result",
            data=img_bytes.getvalue(),
            file_name=f"detected_{uploaded_file.name}",
            mime="image/png"
        )

        # Show detection details
        st.subheader("📋 Detection Details")
        detection_data = get_detection_data(result)
        if detection_data:
            st.dataframe(detection_data)
        else:
            st.info("No tumors detected in this image.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Developed by Shahab | Powered by YOLOv11 Brain Tumor Detection 🧠")
