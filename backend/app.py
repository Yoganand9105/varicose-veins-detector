from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
import cv2

app = FastAPI()

# Load your model
model = tf.keras.models.load_model("varicose_veins_detector.h5")

@app.get("/")
def home():
    return {"message": "Varicose Veins Detection API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img_array = np.expand_dims(img, axis=0)

    prediction = model.predict(img_array)

    score = float(prediction[0][0])

    diagnosis = "Varicose Veins Detected" if score > 0.5 else "Normal"

    return {
        "prediction_score": score,
        "diagnosis": diagnosis
    }
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
import tensorflow as tf
import numpy as np
import cv2

def generate_heatmap(img_array, model, last_conv_layer_name = "Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))

    return heatmap