import io
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI(title="Pneumonia Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Sesuaikan nama model dengan hasil training 2 kelas terbaru
MODEL_PATH = "model_pneumonia_final.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model {MODEL_PATH} berhasil dimuat!")
except Exception as e:
    print(f"⚠️ Peringatan: Model belum ditemukan. Pastikan Anda sudah menjalankan train_local.py")

# 2. Kelas disederhanakan menjadi 2 (Urutan abjad: N lalu P)
CLASSES = ["Normal", "Pneumonia"]

# 3. Fungsi CLAHE diaktifkan kembali agar persis dengan kondisi training
def apply_clahe_for_inference(image_pil):
    img_array = np.array(image_pil)
    img_uint8 = np.clip(img_array, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl_img = clahe.apply(gray)
    rgb_img = cv2.cvtColor(cl_img, cv2.COLOR_GRAY2RGB)
    return rgb_img

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))
        
        # Terapkan CLAHE dan Normalisasi (1./255)
        processed_img = apply_clahe_for_inference(image)
        img_array = processed_img / 255.0 
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prediksi
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0])) * 100
        
        return {
            "status": CLASSES[class_idx],
            "confidence": round(confidence, 2)
        }
        
    except Exception as e:
        return {"error": str(e)}