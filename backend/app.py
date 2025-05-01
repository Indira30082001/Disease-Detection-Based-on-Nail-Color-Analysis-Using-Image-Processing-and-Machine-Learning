from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import logging
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

# Configure CORS
CORS(app, 
     resources={r"/*": {
         "origins": ["http://localhost:3001"],
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization"],
         "supports_credentials": True,
         "max_age": 3600
     }})

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define expected input shape
TARGET_SIZE = (128, 128)

# Get absolute path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model', 'disease_detection_model.h5')

# Global variable for the model
model = None

def load_model_from_path():
    """Load the model from the specified path."""
    global model
    try:
        logger.info(f"Attempting to load model from: {model_path}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            return False
            
        # Load the model with custom object scope to ensure proper loading
        with tf.keras.utils.custom_object_scope({'tf': tf}):
            model = load_model(model_path)
        
        # Compile with proper settings
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Warm up the model with a dummy prediction
        dummy_input = np.zeros((1, 128, 128, 3))
        _ = model.predict(dummy_input)
        
        logger.info("Model loaded and compiled successfully")
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shape: {model.output_shape}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Model path: {model_path}")
        return False

# Load the model when the application starts
if not load_model_from_path():
    logger.error("Failed to load model during initialization")

def analyze_image_features(img_array):
    """Analyze image features including color and texture."""
    # Calculate color features
    mean_color = np.mean(img_array, axis=(0,1))
    std_color = np.std(img_array, axis=(0,1))
    
    # Calculate blue dominance
    blue_value = float(mean_color[2])
    red_value = float(mean_color[0])
    green_value = float(mean_color[1])
    
    # Calculate various color metrics
    is_bluish = blue_value > (red_value * 1.1) and blue_value > (green_value * 1.1)
    blue_ratio = blue_value / (red_value + green_value + 1e-7)
    
    # Calculate texture features
    color_variance = float(np.var(img_array, axis=(0,1)).mean())
    texture_energy = float(np.sum(img_array ** 2) / img_array.size)
    
    # Calculate edge features
    dx = img_array[:, 1:] - img_array[:, :-1]
    dy = img_array[1:, :] - img_array[:-1, :]
    edge_strength = float(np.sqrt(np.mean(dx**2) + np.mean(dy**2)))
    
    # Calculate curvature (for clubbing detection)
    height, width = img_array.shape[:2]
    center_profile = img_array[height//2, :, 0]
    curvature = np.gradient(np.gradient(center_profile))
    max_curvature = float(np.max(np.abs(curvature)))
    
    # Calculate symmetry
    left_half = img_array[:, :width//2]
    right_half = np.flip(img_array[:, width//2:], axis=1)
    symmetry_score = float(1.0 - np.mean(np.abs(left_half - right_half)))
    
    # Calculate additional features for other diseases
    # For Melanoma detection
    color_irregularity = float(np.std(mean_color) / np.mean(mean_color))
    border_irregularity = float(np.mean(np.abs(dx)) + np.mean(np.abs(dy)))
    
    # For Onychogryphosis detection
    vertical_asymmetry = float(np.mean(np.abs(img_array[:, :width//2] - img_array[:, width//2:])))
    thickness_variation = float(np.std(np.mean(img_array, axis=(0,2))))
    
    # For Pitting detection
    local_variations = float(np.mean(np.abs(img_array[1:] - img_array[:-1])))
    surface_roughness = float(np.std(img_array, axis=(0,1,2)))
    
    logger.info(f"Color analysis - R: {red_value:.2f}, G: {green_value:.2f}, B: {blue_value:.2f}")
    logger.info(f"Blue ratio: {blue_ratio:.2f}, Is bluish: {is_bluish}")
    logger.info(f"Color variance: {color_variance:.2f}, Texture energy: {texture_energy:.2f}")
    logger.info(f"Edge strength: {edge_strength:.2f}, Max curvature: {max_curvature:.2f}")
    logger.info(f"Symmetry score: {symmetry_score:.2f}")
    logger.info(f"Color irregularity: {color_irregularity:.2f}, Border irregularity: {border_irregularity:.2f}")
    logger.info(f"Vertical asymmetry: {vertical_asymmetry:.2f}, Thickness variation: {thickness_variation:.2f}")
    logger.info(f"Local variations: {local_variations:.2f}, Surface roughness: {surface_roughness:.2f}")
    
    return {
        'is_bluish': bool(is_bluish),
        'blue_ratio': float(blue_ratio),
        'color_variance': float(color_variance),
        'texture_energy': float(texture_energy),
        'edge_strength': float(edge_strength),
        'max_curvature': float(max_curvature),
        'symmetry_score': float(symmetry_score),
        'mean_colors': [float(x) for x in mean_color],
        'std_colors': [float(x) for x in std_color],
        'color_irregularity': float(color_irregularity),
        'border_irregularity': float(border_irregularity),
        'vertical_asymmetry': float(vertical_asymmetry),
        'thickness_variation': float(thickness_variation),
        'local_variations': float(local_variations),
        'surface_roughness': float(surface_roughness)
    }

def detect_clubbing_characteristics(features):
    """Detect specific characteristics of nail clubbing."""
    # Clubbing typically shows:
    # 1. High curvature at the nail tip
    # 2. Good symmetry
    # 3. Moderate to high edge strength
    # 4. Less color variation
    
    curvature_score = min(1.0, features['max_curvature'] / 0.5)  # Normalize curvature
    symmetry_weight = features['symmetry_score']
    edge_weight = min(1.0, features['edge_strength'] / 0.4)
    color_uniformity = 1.0 - min(1.0, features['color_variance'] / 0.2)
    
    # Combine scores with weights
    clubbing_score = (
        0.4 * curvature_score +
        0.3 * symmetry_weight +
        0.2 * edge_weight +
        0.1 * color_uniformity
    )
    
    return float(clubbing_score)

def adjust_predictions(predictions, features):
    """Adjust predictions based on image features with balanced weights."""
    adjusted = predictions.copy()
    
    # Class indices
    MELANOMA_IDX = 0      # Index for Acral_Lentiginous_Melanoma
    BLUE_FINGER_IDX = 1   # Index for blue_finger class
    CLUBBING_IDX = 2      # Index for clubbing class
    HEALTHY_NAIL_IDX = 3  # Index for healthy nail class
    ONYCHOGRYPHOSIS_IDX = 4  # Index for Onychogryphosis
    PITTING_IDX = 5       # Index for pitting
    
    try:
        # Initialize adjustment factors
        melanoma_factor = 1.0
        blue_factor = 1.0
        clubbing_factor = 1.0
        healthy_factor = 1.0
        onychogryphosis_factor = 1.0
        pitting_factor = 1.0
        
        # Calculate clubbing characteristics
        clubbing_score = detect_clubbing_characteristics(features)
        logger.info(f"Clubbing detection score: {clubbing_score:.3f}")
        
        # Melanoma detection adjustments
        if features['color_irregularity'] > 0.3 and features['border_irregularity'] > 0.4:
            melanoma_factor *= 1.5  # Increased boost
            healthy_factor *= 0.7   # More aggressive reduction
        elif features['color_irregularity'] > 0.2:  # Less strict threshold
            melanoma_factor *= 1.2
        
        # Blue finger adjustments
        if features['is_bluish'] and features['blue_ratio'] > 0.4:
            blue_boost = min(1.5, 1.0 + (float(features['blue_ratio']) - 0.33))
            blue_factor *= blue_boost
            if clubbing_score < 0.5:
                clubbing_factor *= 0.8  # More aggressive reduction
        else:
            blue_factor *= 0.8  # More aggressive reduction
        
        # Clubbing adjustments - More strict criteria
        if clubbing_score > 0.7:  # Increased threshold
            clubbing_factor *= (1.0 + clubbing_score * 0.4)  # Increased boost
            blue_factor *= 0.8
            healthy_factor *= 0.8
        elif clubbing_score < 0.4:  # More strict threshold
            clubbing_factor *= 0.7  # More aggressive reduction
            healthy_factor *= 1.2
        
        # Onychogryphosis adjustments
        if features['vertical_asymmetry'] > 0.4 and features['thickness_variation'] > 0.3:
            onychogryphosis_factor *= 1.5  # Increased boost
            healthy_factor *= 0.7
        elif features['vertical_asymmetry'] > 0.3:  # Less strict threshold
            onychogryphosis_factor *= 1.2
        
        # Pitting adjustments
        if features['local_variations'] > 0.3 and features['surface_roughness'] > 0.4:
            pitting_factor *= 1.5  # Increased boost
            healthy_factor *= 0.7
        elif features['local_variations'] > 0.2:  # Less strict threshold
            pitting_factor *= 1.2
        
        # Apply adjustments
        adjusted[0, MELANOMA_IDX] *= melanoma_factor
        adjusted[0, BLUE_FINGER_IDX] *= blue_factor
        adjusted[0, CLUBBING_IDX] *= clubbing_factor
        adjusted[0, HEALTHY_NAIL_IDX] *= healthy_factor
        adjusted[0, ONYCHOGRYPHOSIS_IDX] *= onychogryphosis_factor
        adjusted[0, PITTING_IDX] *= pitting_factor
        
        # Log adjustment factors
        logger.info(f"Adjustment factors - Melanoma: {melanoma_factor:.2f}, Blue: {blue_factor:.2f}, "
                   f"Clubbing: {clubbing_factor:.2f}, Healthy: {healthy_factor:.2f}, "
                   f"Onychogryphosis: {onychogryphosis_factor:.2f}, Pitting: {pitting_factor:.2f}")
        
        # Ensure no negative values
        adjusted = np.maximum(adjusted, 0)
        
        # Normalize probabilities
        adjusted = adjusted / np.sum(adjusted)
        
        # Apply minimum confidence threshold of 60%
        min_confidence = 0.6
        adjusted = np.where(adjusted < min_confidence, 0, adjusted)
        
        # If all predictions are below threshold, keep the highest one
        if np.sum(adjusted) == 0:
            max_idx = np.argmax(predictions[0])
            adjusted[0, max_idx] = predictions[0, max_idx]
        
        # Renormalize after thresholding
        adjusted = adjusted / np.sum(adjusted)
        
        # Log adjustments
        logger.info("Prediction adjustments:")
        disease_classes = [
            "Acral_Lentiginous_Melanoma",
            "blue_finger",
            "clubbing",
            "Healthy_Nail",
            "Onychogryphosis",
            "pitting"
        ]
        for i, (orig, adj) in enumerate(zip(predictions[0], adjusted[0])):
            logger.info(f"{disease_classes[i]}: {orig:.3f} -> {adj:.3f}")
        
        return adjusted
    except Exception as e:
        logger.error(f"Error adjusting predictions: {str(e)}")
        return predictions

@app.route('/')
def index():
    """Check if the model is loaded and API is running."""
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({
        "status": "API is running",
        "model_status": model_status,
        "model_path": model_path
    })

def prepare_image(image_bytes):
    """Prepare image for prediction."""
    try:
        # Open and convert image to RGB
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        logger.info(f"Original image size: {img.size}")
        
        # Resize image to target size using LANCZOS resampling
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize to [0, 1] range
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Log image statistics
        logger.info(f"Image statistics - Min: {img_array.min():.3f}, Max: {img_array.max():.3f}, Mean: {img_array.mean():.3f}")
        logger.info(f"Processed image shape: {img_array.shape}")
        
        return img_array
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return None

def analyze_nail_features(img_array):
    """Analyze nail features to help with prediction."""
    try:
        # Remove batch dimension for analysis
        img = img_array[0]
        
        # Calculate basic statistics
        mean_color = np.mean(img, axis=(0,1))
        std_color = np.std(img, axis=(0,1))
        
        # Calculate color ratios
        max_color = np.max(mean_color)
        color_ratios = mean_color / (max_color + 1e-7)
        
        # Calculate texture features
        texture_energy = np.mean(np.square(img))
        color_variance = np.mean(std_color)
        
        logger.info(f"Color ratios: R={color_ratios[0]:.3f}, G={color_ratios[1]:.3f}, B={color_ratios[2]:.3f}")
        logger.info(f"Texture energy: {texture_energy:.3f}, Color variance: {color_variance:.3f}")
        
        return {
            'color_ratios': color_ratios,
            'texture_energy': texture_energy,
            'color_variance': color_variance
        }
    except Exception as e:
        logger.error(f"Error analyzing features: {str(e)}")
        return None

def boost_predictions(predictions, min_confidence=0.5):
    """Boost prediction confidences while maintaining relative relationships."""
    # Get the highest confidence
    max_conf = np.max(predictions)
    
    if max_conf < min_confidence:
        # Calculate boost factor needed to get max prediction to target confidence
        boost_factor = min_confidence / max_conf
        
        # Apply boosting with softmax to maintain probability distribution
        boosted = predictions * boost_factor
        # Apply softmax to normalize and enhance differences
        exp_preds = np.exp(boosted - np.max(boosted))
        boosted = exp_preds / np.sum(exp_preds)
        
        return boosted
    return predictions

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Handle prediction requests."""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        # Check if model is loaded
        if model is None:
            if not load_model_from_path():
                return jsonify({
                    'error': 'Model not loaded and failed to load. Please check server logs.'
                }), 500

        # Check for image in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        # Read and process image
        image = request.files['image'].read()
        processed = prepare_image(image)
        
        if processed is None:
            return jsonify({'error': 'Image processing failed'}), 400

        # Make prediction
        prediction = model.predict(processed, verbose=0)
        
        # Get class names
        disease_classes = [
            "Acral_Lentiginous_Melanoma",
            "blue_finger",
            "clubbing",
            "Healthy_Nail",
            "Onychogryphosis",
            "pitting"
        ]
        
        # Create predictions list
        predictions_with_confidence = []
        for i, prob in enumerate(prediction[0]):
            predictions_with_confidence.append({
                'disease': disease_classes[i],
                'confidence': float(prob)
            })
        
        # Sort predictions by confidence
        predictions_with_confidence.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Log predictions
        logger.info("Predictions:")
        for pred in predictions_with_confidence:
            logger.info(f"{pred['disease']}: {pred['confidence']*100:.2f}%")
        
        return jsonify({
            'success': True,
            'disease': predictions_with_confidence[0]['disease'],
            'confidence': float(predictions_with_confidence[0]['confidence']),
            'all_predictions': predictions_with_confidence
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
