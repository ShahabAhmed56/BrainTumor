import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

# ----------------------------
# Streamlit Config (must be first Streamlit command!)
# ----------------------------
st.set_page_config(page_title="YOLOv11 Brain Tumor Detector", page_icon="🧠", layout="wide")

# ----------------------------
# Load YOLOv11 Model
# ----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # path to your trained YOLOv11 model

model = load_model()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🧠 Brain Tumor Detection using YOLOv11")
st.write("Upload MRI scan images to detect **Pituitary, No Tumor, Meningioma, Glioma**.")

uploaded_files = st.file_uploader(
    "Choose one or more MRI images...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"📤 Uploaded: {uploaded_file.name}", use_column_width=True)

        # Run inference
        with st.spinner("Detecting tumors..."):
            results = model(image)
        result = results[0]

        # Plot result
        plotted_img = result.plot()

        st.subheader(f"🔍 Detection Result: {uploaded_file.name}")
        st.image(plotted_img, caption="Detected Image", use_column_width=True)

        # Save output image to memory
        img_bytes = io.BytesIO()
        Image.fromarray(plotted_img).save(img_bytes, format="PNG")

        # Download button
        st.download_button(
            label="💾 Download Result",
            data=img_bytes.getvalue(),
            file_name=f"detected_{uploaded_file.name}",
            mime="image/png"
        )

        # Show detection details
        st.subheader("📋 Detection Details")
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            data = []
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
            st.dataframe(data)
        else:
            st.info("No tumors detected in this image.")
