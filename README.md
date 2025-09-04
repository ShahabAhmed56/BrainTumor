# 🧠 YOLOv11 Brain Tumor Detection App

This Streamlit app uses a custom-trained YOLOv11 model to detect brain tumors from MRI scans. It supports detection of **Pituitary**, **Meningioma**, **Glioma**, and **No Tumor** classes. Users can upload one or more images, view detection results, and download annotated outputs directly from the browser.

---

## 🚀 Features

- Upload multiple MRI images (JPG, JPEG, PNG)
- Run YOLOv11 inference on each image
- View bounding boxes and class labels
- Download annotated images
- View detection details (class, confidence, coordinates)

---

## 🧰 Tech Stack

- Python 3.11
- Streamlit 1.39.0
- Ultralytics YOLOv11 (`ultralytics==8.3.78`)
- Pillow for image handling
- OpenCV (headless) for drawing boxes
- NumPy for image preprocessing

---

## 📦 Installation (Local)

```bash
git clone https://github.com/ShahabAhmed56/BrainTumor.git
cd braintumor
pip install -r requirements.txt
streamlit run app.py

