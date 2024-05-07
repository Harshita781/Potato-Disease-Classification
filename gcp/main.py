from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

model = None
interpreter = None
input_index = None
output_index = None

BUCKET_NAME = "pdc-tf-model-1"

class_names = ["Early Blight", "Late Blight", "Healthy"]


def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)


def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/potatoes.h5",
        )
        model = tf.keras.models.load_model("/tmp/potatoes.h5", compile=False)

    image = request.files["file"]

    image = np.array(
        Image.open(image).convert("RGB").resize((256, 256))
    )

    image = image/255

    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)

    print("Predictions: ", predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return {"class": predicted_class, "confidence": confidence}
