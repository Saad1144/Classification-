import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- CONFIGURATION ---
# Chemins vers tes dossiers d'images
TRAIN_DIR = '../dataset/train'
VAL_DIR = '../dataset/validation'

# Tes catégories de déchets (doivent correspondre aux noms des dossiers)
CATEGORIES = sorted(os.listdir(TRAIN_DIR)) 

print(f"Catégories détectées automatiquement : {CATEGORIES}")
print(f"Nombre de classes : {len(CATEGORIES)}")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5  # Augmente à 10-20 pour de meilleurs résultats (plus long)

print("--- 1. Préparation des données ---")

# Création de données augmentées (Régularisation pour éviter l'overfitting)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Générateurs
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

print("--- 2. Construction du modèle (Transfer Learning) ---")

# On utilise MobileNetV2 pré-entraîné sur ImageNet
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # On fige les couches de base

# On ajoute nos propres couches de classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x) # Régularisation
predictions = Dense(len(CATEGORIES), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("--- 3. Entraînement ---")
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# Sauvegarde du modèle
model.save('eco_model.h5')
print(f"Modèle sauvegardé sous 'eco_model.h5' avec {len(CATEGORIES)} classes.")