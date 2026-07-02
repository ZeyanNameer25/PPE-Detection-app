# 🦺 PPE Detection using YOLO

A real-time Personal Protective Equipment (PPE) detection web application built using **Ultralytics YOLO**, **Streamlit**, and **OpenCV**. The application detects workplace safety equipment from **images**, **videos**, and **live webcam streams** using a custom-trained deep learning model.

---

## Features

- 📷 Image Detection
- 🎥 Video Detection
- 📹 Real-Time Webcam Detection
- 🎯 Adjustable Confidence Threshold
- 🎯 Adjustable IoU Threshold
- 📊 Live Detection Statistics
- ⚡ Fast YOLO Inference
- ☁️ Streamlit Cloud Ready
- 🧠 Powered by a custom-trained YOLO model

---

## PPE Classes

The model detects the following Personal Protective Equipment (PPE):

- 🪖 Helmet
- 😷 Mask
- 🦺 Safety Vest
- 🥾 Boots
- 🧤 Gloves

---

## Tech Stack

- Python
- Streamlit
- Ultralytics YOLO
- PyTorch
- OpenCV
- Pillow (PIL)
- Streamlit-WebRTC
- AV
- NumPy

---

## Project Structure

```text
ppe-detection-app/
│
├── app.py
├── ppe.pt
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ppe-detection-app.git
cd ppe-detection-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

The application will automatically open in your default web browser.

---

## Model

This project uses a custom-trained **Ultralytics YOLO** model (`ppe.pt`) for detecting Personal Protective Equipment in images, videos, and live webcam feeds.

---

## Deployment

This application is designed to be deployed on **Streamlit Cloud**.

---

## Future Improvements

- Detection confidence history
- FPS monitoring
- Screenshot capture
- Object tracking
- PPE compliance analytics
- Multi-camera support

---

## Author

**Zeyan Nameer**