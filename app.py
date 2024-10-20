import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import cv2
from torchvision import models, transforms

# Step 1: Load a pre-trained DeepLabV3 model for segmentation
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# Preprocessing function to transform the input image
def preprocess(image_pil):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image_pil).unsqueeze(0), image_pil

# Function to generate the mask for clothes
def generate_clothes_mask(image_tensor):
    # Forward pass through the model
    with torch.no_grad():
        output = model(image_tensor)['out'][0]
    output_predictions = output.argmax(0)  # Get the highest scoring category for each pixel

    # Category 15 is the "person" label, which includes clothes
    mask = (output_predictions == 15).byte().cpu().numpy()
    return mask

# Function to exclude hair and skin using color filtering
def refine_clothes_mask(mask, image_pil):
    # Convert PIL image to numpy array for OpenCV
    image_np = np.array(image_pil)

    # Convert to HSV for skin and hair removal
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    # Define skin color range in HSV and create a mask for skin
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Invert the skin mask to get non-skin regions
    non_skin_mask = cv2.bitwise_not(skin_mask)

    # Combine the clothes mask with the non-skin mask to refine the mask
    refined_mask = cv2.bitwise_and(mask, mask, mask=non_skin_mask)
    
    return refined_mask

# Streamlit App
st.title("Advanced Clothes Masking App for Stable Diffusion")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image using PIL and convert to RGB
    image_pil = Image.open(uploaded_file).convert("RGB")
    
    # Preprocess the image for DeepLabV3 model
    image_tensor, image_pil = preprocess(image_pil)
    
    # Generate clothes mask
    clothes_mask = generate_clothes_mask(image_tensor)
    
    # Refine the mask by removing skin and hair
    refined_mask = refine_clothes_mask(clothes_mask, image_pil)
    
    # Display the original image and refined mask
    st.image(image_pil, caption='Uploaded Image', use_column_width=True)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(refined_mask, cmap='gray')
    plt.axis('off')
    
    # Convert the mask to an image for display
    st.image(refined_mask, caption="Clothes Mask", use_column_width=True)

    # Option to download the refined mask
    result_img = Image.fromarray((refined_mask * 255).astype(np.uint8))
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    
    st.download_button(
        label="Download Clothes Mask",
        data=byte_im,
        file_name="clothes_mask.png",
        mime="image/png"
    )
