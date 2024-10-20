import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np
import os
from torchvision import models, transforms
import matplotlib.pyplot as plt

# Step 1: Load the DeepLabV3 segmentation model
model = models.segmentation.deeplabv3_resnet101(weights="COCO_WITH_VOC_LABELS_V1")
model.eval()

# Step 2: Preprocess the uploaded image for the DeepLab model
def preprocess(image):
    preprocess_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess_transform(image).unsqueeze(0)

# Step 3: Generate segmentation mask and filter for clothes
def generate_clothes_mask(image_tensor, image_pil):
    with torch.no_grad():
        output = model(image_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()

    # Filter out everything except the "person" class (category 15 in DeepLabV3)
    person_mask = (output_predictions == 15).astype(np.uint8) * 255

    # Convert the mask to a format usable by OpenCV
    person_mask_cv = np.array(person_mask, dtype=np.uint8)

    # Exclude non-clothing areas (use HSV or other methods)
    original_image_cv = np.array(image_pil)[:, :, ::-1]  # Convert PIL to OpenCV BGR format
    hsv_image = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for typical clothing colors (adjust as needed)
    lower_clothes_hsv = np.array([0, 30, 60])
    upper_clothes_hsv = np.array([179, 255, 255])

    clothes_mask_hsv = cv2.inRange(hsv_image, lower_clothes_hsv, upper_clothes_hsv)

    # Combine the person mask with the clothes color mask to get a clean clothing mask
    combined_mask = cv2.bitwise_and(person_mask_cv, clothes_mask_hsv)

    return combined_mask

# Step 4: Create a downloadable mask as a PNG
def create_downloadable_mask(mask):
    mask_pil = Image.fromarray(mask)
    return mask_pil

# Streamlit app interface
st.title("Clothes Masking App")

# Step 5: Upload the image
uploaded_image = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Convert uploaded image to PIL format
    image_pil = Image.open(uploaded_image).convert("RGB")
    
    # Display the uploaded image
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    # Step 6: Preprocess the image and generate the mask
    image_tensor = preprocess(image_pil)
    clothes_mask = generate_clothes_mask(image_tensor, image_pil)

    # Step 7: Display the mask
    st.write("Generated Clothes Mask")
    st.image(clothes_mask, caption="Clothes Mask", use_column_width=True, clamp=True)

    # Step 8: Create a downloadable version of the mask
    downloadable_mask = create_downloadable_mask(clothes_mask)
    st.write("Download the generated mask:")
    
    # Step 9: Provide a download button for the mask
    mask_download = st.download_button(
        label="Download Mask",
        data=downloadable_mask.tobytes(),
        file_name="clothes_mask.png",
        mime="image/png"
    )
