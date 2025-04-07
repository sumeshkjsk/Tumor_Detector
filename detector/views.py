from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the model once when the server starts
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'Brain_Tumor_Model.h5')
model = load_model(MODEL_PATH)

def predict_tumor(request):
    prediction = None
    image_url = None

    if request.method == 'POST' and request.FILES.get('image'):
        # Save the uploaded image
        img_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(img_file.name, img_file)
        full_path = fs.path(filename)
        image_url = fs.url(filename)

        # Preprocess the image
        img = image.load_img(full_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make prediction
        prediction_prob = model.predict(img_array)[0][0]
        prediction = 'Tumor Detected' if prediction_prob > 0.5 else 'No Tumor Detected'

    return render(request, 'detector/upload.html', {
        'prediction': prediction,
        'image_url': image_url,
    })
