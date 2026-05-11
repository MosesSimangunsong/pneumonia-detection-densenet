import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# ========================================================
# 1. KONFIGURASI PATH WINDOWS LOKAL
# ========================================================
base_dir = r"D:\Semester6\ML\Proyek\Rontgen_AI\dataset\archive\chest_xray\chest_xray"
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

classes = ['NORMAL', 'PNEUMONIA']

# ========================================================
# 2. FUNGSI PREPROCESSING CLAHE
# ========================================================
def apply_clahe(img):
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    img_clahe = cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2RGB)
    return np.float32(img_clahe)

# ========================================================
# 3. GENERATOR DATASET
# ========================================================
print("Memuat dataset dari folder lokal...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=apply_clahe,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False 
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=apply_clahe
)

train_data = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
val_data = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

classes_idx = train_data.classes
weights = compute_class_weight('balanced', classes=np.unique(classes_idx), y=classes_idx)
class_weights = dict(enumerate(weights))

# ========================================================
# 4. ARSITEKTUR MODEL (DENSENET121 - SIGMOID TRICK)
# ========================================================
print("\nMembangun arsitektur DenseNet121...")
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

# UBAH 1: Softmax menjadi Sigmoid agar probabilitas mandiri
predictions = Dense(2, activation='sigmoid')(x) 

model = Model(inputs=base_model.input, outputs=predictions)

from tensorflow.keras.optimizers import Adam
# UBAH 2: Categorical crossentropy menjadi Binary crossentropy
model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

# ========================================================
# 5. CUSTOM CALLBACKS & TRAINING
# ========================================================
# Nama model sedikit diubah agar Anda tahu ini adalah versi Sigmoid
model_save_path = os.path.join(r"D:\Semester6\ML\Proyek\Rontgen_AI\backend-pneumonia", "model_pneumonia_final.h5")

# ModelCheckpoint (Menyimpan versi terbaik selama 100 epoch)
checkpoint_saver = ModelCheckpoint(
    filepath=model_save_path,
    monitor='val_accuracy',
    save_best_only=True, 
    mode='max',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.00001)

print("\n🚀 Mulai melatih model dengan Trik Sigmoid (100 Epochs Paksa)...")
history = model.fit(
    train_data,
    epochs=100,
    steps_per_epoch=100,
    validation_data=val_data,
    class_weight=class_weights,
    callbacks=[checkpoint_saver, reduce_lr] 
)

# ========================================================
# 6. PENYELESAIAN
# ========================================================
print(f"\n✅ SUKSES! Training 100 epoch selesai. Model tersimpan di: {model_save_path}")