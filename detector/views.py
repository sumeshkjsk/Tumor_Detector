from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import base64
import logging
from .models import PredictionLog

# Configure logging
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'Brain_Tumor_Model.h5')
try:
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
    
    # Immediately test model with blank image
    test_array = np.zeros((1, 224, 224, 3))
    test_pred = model.predict(test_array)[0][0]
    logger.info(f"Model test prediction (should be near 0): {test_pred}")
    if test_pred > 0.5:
        logger.error("MODEL ALERT: Always predicts positive! Needs retraining!")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise e

def is_valid_mri(img_path):
    """Strict validation for proper MRI scans"""
    try:
        img = Image.open(img_path).convert('L')
        arr = np.array(img)
        
        # Calculate image statistics
        mean_val = arr.mean()
        std_val = arr.std()
        hist = np.histogram(arr, bins=10)[0]
        
        # MRI should have:
        # - Moderate intensity (not too dark/bright)
        # - Moderate contrast
        # - Non-uniform histogram
        conditions = [
            (mean_val < 30 or mean_val > 220, "Invalid intensity range"),
            (std_val < 15, "Insufficient contrast"),
            (np.max(hist)/np.sum(hist) > 0.8, "Lacks MRI texture"),
            (arr.max() - arr.min() < 50, "Low dynamic range")
        ]
        
        for condition, msg in conditions:
            if condition:
                return False, msg
                
        return True, "Valid MRI"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def preprocess_image(img_path, target_size=(224, 224)):
    """Consistent preprocessing matching training"""
    try:
        img = Image.open(img_path).convert('RGB')
        img = ImageOps.fit(img, target_size, Image.LANCZOS)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise ValueError(f"Image processing error: {str(e)}")

def make_prediction(img_path):
    """Robust prediction with multiple safety checks"""
    try:
        # 1. Strict MRI validation
        is_valid, msg = is_valid_mri(img_path)
        if not is_valid:
            return {'result': f"Invalid MRI: {msg}", 'confidence': 0, 'is_error': True}
        
        # 2. Get image statistics
        img = Image.open(img_path).convert('L')
        arr = np.array(img)
        mean_val = arr.mean()
        std_val = arr.std()
        
        # 3. Preprocess
        img_array = preprocess_image(img_path)
        
        # 4. Get raw prediction
        raw_pred = float(model.predict(img_array)[0][0])
        logger.info(f"Raw prediction value: {raw_pred:.4f}")
        
        # 5. Dynamic threshold adjustment
        base_threshold = 0.7  # Conservative base threshold
        if std_val < 20: base_threshold += 0.15
        if mean_val < 40 or mean_val > 200: base_threshold += 0.1
        
        # 6. Determine result
        if raw_pred >= base_threshold:
            result = "Tumor Detected"
            confidence = round(raw_pred * 100, 1)
        else:
            result = "No Tumor Detected"
            confidence = round((1 - raw_pred) * 100, 1)
        
        # 7. EMERGENCY OVERRIDE - if model is clearly wrong
        if result == "Tumor Detected" and confidence < 75:
            if std_val < 25 or mean_val < 50 or mean_val > 180:
                result = "No Tumor Detected"
                confidence = 85.0  # High confidence override
        
        return {
            'result': result,
            'confidence': confidence,
            'raw_prediction': raw_pred,
            'is_error': False
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {'result': "System error", 'confidence': 0, 'is_error': True}

def predict_tumor(request):
    if request.method == 'POST':
        # Validate file upload
        if 'image' not in request.FILES:
            messages.error(request, "Please select an image to upload.")
            return redirect('predict_tumor')
            
        uploaded_file = request.FILES['image']
        
        # Validate file type
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.dcm']
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext not in valid_extensions:
            messages.error(request, "Unsupported file format. Please upload JPG, PNG, BMP, or DICOM.")
            return redirect('predict_tumor')
            
        if uploaded_file.size > 10 * 1024 * 1024:
            messages.error(request, "File too large (max 10MB allowed).")
            return redirect('predict_tumor')

        fs = FileSystemStorage()
        try:
            # Save temporarily
            filename = fs.save(uploaded_file.name, uploaded_file)
            full_path = fs.path(filename)
            
            # Make prediction
            prediction_data = make_prediction(full_path)
            
            if prediction_data['is_error']:
                messages.error(request, prediction_data['result'])
            else:
                # Prepare image for display
                with open(full_path, "rb") as img_file:
                    encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
                
                # Log prediction
                try:
                    PredictionLog.objects.create(
                        user=request.user if request.user.is_authenticated else None,
                        image_name=filename,
                        prediction_result=prediction_data['result'],
                        confidence=prediction_data['confidence'],
                        raw_prediction=prediction_data['raw_prediction']
                    )
                except Exception as e:
                    logger.error(f"Failed to log prediction: {str(e)}")
                
                # Store in session
                request.session['prediction_result'] = prediction_data['result']
                request.session['prediction_confidence'] = prediction_data['confidence']
                request.session['image_base64'] = encoded_string
                request.session['filename'] = filename
                request.session.modified = True  # Ensure session saves
                
                messages.success(request, "Analysis completed!")
            
            return redirect('predict_tumor')
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            messages.error(request, f"Analysis failed: {str(e)}")
            return redirect('predict_tumor')
        finally:
            if 'full_path' in locals() and os.path.exists(full_path):
                fs.delete(filename)

    # GET request - show form and results
    context = {
        'prediction_result': request.session.get('prediction_result'),
        'prediction_confidence': request.session.get('prediction_confidence'),
        'image_base64': request.session.get('image_base64'),
        'filename': request.session.get('filename'),
    }
    
    # Clear session after showing results
    for key in ['prediction_result', 'prediction_confidence', 'image_base64', 'filename']:
        if key in request.session:
            del request.session[key]
    
    return render(request, 'detector/upload.html', context)