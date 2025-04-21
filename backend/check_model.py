import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Load the model
    model = tf.keras.models.load_model('backend/model/disease_detection_model.h5')
    
    # Print model information
    logger.info("Model loaded successfully")
    logger.info(f"Input shape: {model.layers[0].input_shape}")
    logger.info("\nModel summary:")
    model.summary()
    
except Exception as e:
    logger.error(f"Error loading model: {str(e)}") 