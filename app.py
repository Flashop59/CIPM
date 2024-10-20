import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to apply the GrabCut algorithm for clothes segmentation
def grabcut_clothes_segmentation(image_pil):
    # Convert PIL image to OpenCV format (BGR)
    image = np.array(image_pil)[:, :, ::-1]

    # Create an initial mask for GrabCut
    mask = np.zeros(image.shape[:2], np.uint8)

    # Define background and foreground models for GrabCut
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Define a rectangle around the region where clothes are likely to appear
    height, width = image.shape[:2]
    rect = (int(width * 0.1), int(height * 0.2), int(width * 0.9), int(height * 0.9))  # Exclude top 20% (likely head)

    # Apply the GrabCut algorithm
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Create a binary mask where 0 and 2 are background, and 1 and 3 are foreground (clothes)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Use the mask to extract the clothes from the image
    result = image * mask2[:, :, np.newaxis]

    return result

# Streamlit interface
st.title("Clothes Segmentation App (GrabCut)")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Convert the uploaded image to PIL format
    image_pil = Image.open(uploaded_image).convert("RGB")

    # Display the uploaded image
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    # Apply the GrabCut algorithm for clothes segmentation
    segmented_image = grabcut_clothes_segmentation(image_pil)

    # Convert the segmented result to PIL for display
    result_pil = Image.fromarray(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))

    # Display the result
    st.image(result_pil, caption="Clothes Mask", use_column_width=True)

    # Allow downloading of the masked image
    st.download_button(
        label="Download Masked Image",
        data=result_pil.tobytes(),
        file_name="clothes_mask.png",
        mime="image/png"
    )
