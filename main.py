from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import os

app = FastAPI()

# Configuration CORS pour permettre au frontend de communiquer avec le backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, remplace "*" par l'URL de ton site
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du modèle
print("Chargement du modèle...")
MODEL_PATH = 'eco_model.h5'

# Vérifie si le modèle existe, sinon on prévient l'utilisateur
if not os.path.exists(MODEL_PATH):
    print("ATTENTION: Le fichier 'eco_model.h5' est introuvable. Lancez d'abord train_model.py")
    model = None
else:
    model = load_model(MODEL_PATH)
    # Les catégories doivent être dans le même ordre que lors de l'entraînement (alphabétique généralement)
    CATEGORIES = ['carton', 'compost', 'metal', 'plastique', 'verre'] 

def prepare_image(img_bytes):
    """Transforme l'image reçue en format compréhensible par l'IA"""
    img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 # Normalisation
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(content={"error": "Modèle non entraîné"}, status_code=500)

    try:
        # Lecture de l'image envoyée
        img_bytes = await file.read()
        processed_img = prepare_image(img_bytes)
        
        # Prédiction
        predictions = model.predict(processed_img)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        label = CATEGORIES[predicted_class_index]
        
        return JSONResponse(content={
            "label": label,
            "confidence": round(confidence * 100, 2),
            "message": f"Objet détecté : {label} (Confiance : {round(confidence * 100, 2)}%)"
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Point de départ si on lance le fichier directement
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)