import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

# Function to load image and convert to HSV
def load_image(image_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return image, hsv_image

# Function to generate mask for clothes based on color range
def get_clothes_mask(hsv_image, lower_color, upper_color, no_hair):
    # Create mask for the color range
    clothes_mask = cv2.inRange(hsv_image, lower_color, upper_color)
    
    if no_hair:
        # Assuming hair is typically darker, filter out hair using HSV range
        lower_hair = np.array([0, 0, 0], dtype=np.uint8)  # Adjust for darker shades
        upper_hair = np.array([180, 255, 80], dtype=np.uint8)
        hair_mask = cv2.inRange(hsv_image, lower_hair, upper_hair)
        
        # Invert the hair mask to exclude hair regions
        hair_mask_inv = cv2.bitwise_not(hair_mask)
        clothes_mask = cv2.bitwise_and(clothes_mask, hair_mask_inv)
    
    return clothes_mask

# Function to save and display the mask (in-memory)
def save_mask(mask, original_image):
    # Convert the mask to 3 channels to apply it on the original image
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    masked_image = cv2.bitwise_and(original_image, mask_rgb)
    
    # Save the mask in-memory as a PNG
    _, mask_png = cv2.imencode('.png', mask)
    return masked_image, mask_png

# Streamlit App
st.title("Clothes Masking App for Stable Diffusion")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image from the file uploader
    image = Image.open(uploaded_file)
    image = np.array(image)  # Convert PIL to NumPy array
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Set static HSV color ranges for clothes (you can tweak these)
    lower_color = np.array([0, 50, 50], dtype=np.uint8)  # Example lower bound for red
    upper_color = np.array([10, 255, 255], dtype=np.uint8)  # Example upper bound for red

    # Option to exclude hair from mask
    no_hair = st.checkbox("Exclude Hair from Mask", value=True)
    
    # Mask generation
    if st.button("Generate Mask"):
        clothes_mask = get_clothes_mask(hsv_image, lower_color, upper_color, no_hair)
        
        # Save and display mask
        masked_image, mask_png = save_mask(clothes_mask, image)
        st.image(masked_image, caption='Generated Mask', use_column_width=True)
        
        # Provide download link for the mask (in-memory file)
        st.download_button(
            label="Download Mask",
            data=io.BytesIO(mask_png),
            file_name="clothes_mask.png",
            mime="image/png"
        )
