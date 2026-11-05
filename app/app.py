import tensorflow as tf
import numpy as np
import gradio as gr
from PIL import Image
import os

# Suppress logs and rounding errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the trained model
model = tf.keras.models.load_model('pneumonia_detection.keras')
# Load model without optimizer to avoid the 'rmsprop' warning
#model = tf.keras.models.load_model('pneumonia_detection.keras', compile=False)

# Define the image size the model expects
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Function to preprocess the image and make a prediction
def predict_pneumonia(image):
    if image is None:
        return "Please upload an image."

    # Convert PIL Image to numpy array
    image_array = np.asarray(image.convert('L')) # Convert to grayscale

    # Resize image to the expected dimensions
    image_array = Image.fromarray(image_array).resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.asarray(image_array)

    # Add batch and channel dimensions (for grayscale, it's (1, height, width, 1))
    img_batch = np.expand_dims(image_array, axis=0)
    img_batch = np.expand_dims(img_batch, axis=-1)

    # The model expects normalized pixel values (0-1), which is handled by Rescaling layer in the model
    # So, no manual normalization (e.g., img_batch / 255.0) is needed here if Rescaling layer is in model.

    prediction = model.predict(img_batch)[0][0]

    # Define class labels
    class_names = ['NORMAL', 'PNEUMONIA']

    # Get the predicted class and confidence
    if prediction > 0.5:
        predicted_class = 'PNEUMONIA'
        confidence = prediction
    else:
        predicted_class = 'NORMAL'
        confidence = 1 - prediction

    return f"Prediction: {predicted_class} (Confidence: {confidence:.2f})"

# Set up the Gradio interface
iface = gr.Interface(
    fn=predict_pneumonia,
    inputs=gr.Image(type="pil", label="Upload Chest X-ray Image"),
    outputs="text",
    title="Pneumonia Detection from Chest X-ray",
    description="Upload a chest X-ray image to check for pneumonia. The model will output 'Pneumonia' or 'Normal' along with a confidence score."
)

# Launch the Gradio interface
#iface.launch(share=True)
if __name__ == "__main__":
    # share=False is required for Spaces, and we launch on port 7860
    iface.launch(server_name="0.0.0.0", server_port=7860,share=False)
