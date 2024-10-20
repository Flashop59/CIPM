import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# Step 1: Load the image and convert to HSV
def load_image(image):
    original_image = np.array(image)
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)
    return original_image, hsv_image

# Step 2: Define the color range for clothes and exclude the upper region (hair)
def get_clothes_mask(hsv_image, lower_color, upper_color, hair_exclusion_ratio=0.2):
    # Get image dimensions
    height, width = hsv_image.shape[:2]

    # Define the mask that only includes pixels within the color range
    mask = cv2.inRange(hsv_image, lower_color, upper_color)

    # Exclude the top portion of the mask (which may contain hair) by zeroing it out
    exclusion_height = int(height * hair_exclusion_ratio)
    mask[:exclusion_height, :] = 0  # Exclude top 20% of the image (adjust ratio as needed)

    return mask

# Step 3: Save the mask and convert it to Pillow image for download
def create_downloadable_mask(mask, original_image):
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    masked_image = cv2.bitwise_and(original_image, mask_rgb)
    return Image.fromarray(mask)

# Streamlit app
st.title("Clothes Color Masking App (Excluding Hair)")

# Step 4: Upload Image
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Step 5: Input for selecting HSV color range
    st.write("Select the HSV color range for the clothing")
    lower_h = st.slider("Lower Hue", 0, 180, 0)
    lower_s = st.slider("Lower Saturation", 0, 255, 50)
    lower_v = st.slider("Lower Value", 0, 255, 50)
    upper_h = st.slider("Upper Hue", 0, 180, 180)
    upper_s = st.slider("Upper Saturation", 0, 255, 255)
    upper_v = st.slider("Upper Value", 0, 255, 255)

    # Convert the uploaded image into an HSV format for masking
    original_image, hsv_image = load_image(image)

    # Define the color range for clothes
    lower_color = np.array([lower_h, lower_s, lower_v])
    upper_color = np.array([upper_h, upper_s, upper_v])

    # Step 6: Generate the clothes mask and exclude hair
    clothes_mask = get_clothes_mask(hsv_image, lower_color, upper_color)

    # Display the generated mask
    st.write("Generated Mask (Hair Excluded)")
    st.image(clothes_mask, caption='Clothes Mask', use_column_width=True, clamp=True)

    # Create downloadable mask
    masked_image_pil = create_downloadable_mask(clothes_mask, original_image)

    # Step 7: Provide a download button for the mask
    st.write("Download the generated mask")
    mask_download = st.download_button(
        label="Download Mask",
        data=masked_image_pil.tobytes(),
        file_name="clothes_mask_no_hair.png",
        mime="image/png"
    )
