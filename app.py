import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
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

# Function to save and display the mask
def save_mask(mask, original_image, mask_output_path):
    os.makedirs(os.path.dirname(mask_output_path), exist_ok=True)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    masked_image = cv2.bitwise_and(original_image, mask_rgb)
    cv2.imwrite(mask_output_path, mask)
    return masked_image

# Streamlit App
st.title("Clothes Masking App for Stable Diffusion")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_path = "input_image.png"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display uploaded image
    st.image(Image.open(image_path), caption='Uploaded Image', use_column_width=True)
    
    # Load the image and convert to HSV
    original_image, hsv_image = load_image(image_path)
    
    # Color range selection (in HSV format)
    st.write("Select the color range of the clothes:")
    lower_h = st.slider("Lower Hue", 0, 180, 0)
    lower_s = st.slider("Lower Saturation", 0, 255, 50)
    lower_v = st.slider("Lower Value", 0, 255, 50)
    upper_h = st.slider("Upper Hue", 0, 180, 180)
    upper_s = st.slider("Upper Saturation", 0, 255, 255)
    upper_v = st.slider("Upper Value", 0, 255, 255)
    
    lower_color = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
    upper_color = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)
    
    # Option to exclude hair from mask
    no_hair = st.checkbox("Exclude Hair from Mask", value=True)
    
    # Mask generation
    if st.button("Generate Mask"):
        clothes_mask = get_clothes_mask(hsv_image, lower_color, upper_color, no_hair)
        mask_output_path = "clothes_mask.png"
        
        # Save and display mask
        masked_image = save_mask(clothes_mask, original_image, mask_output_path)
        st.image(masked_image, caption='Generated Mask', use_column_width=True)
        
        # Provide download link for the mask
        with open(mask_output_path, "rb") as f:
            st.download_button("Download Mask", f, file_name="clothes_mask.png", mime="image/png")
