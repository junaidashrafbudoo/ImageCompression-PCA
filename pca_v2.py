import numpy as np  
import cv2  
import streamlit as st 
from sklearn.decomposition import PCA  
import os, random, time  

from utils import get_image_list, load_image, apply_pca

st.set_page_config(layout="wide")

# Check if 'random_file' is in session state, initialize it if not
if 'random_file' not in st.session_state:
    st.session_state.random_file = "cat2.jpg"

st.title("Principal Component Analysis (PCA) for Image Compression")

# Sidebar with instructions and file upload
st.sidebar.header("Upload or Select an Image")
st.sidebar.write("Upload an image or select an example image to see PCA in action.")

uploaded_file = st.sidebar.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg', 'tiff', 'bmp'], label_visibility="collapsed")

if uploaded_file is not None:
    st.sidebar.info(f"File uploaded: {uploaded_file.name}")
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_image_size_kb = len(file_bytes) / 1024  # Calculate original image size
else:
    if st.sidebar.button("Try New Example Image!"):
        random_file = random.choice(get_image_list())
        st.session_state.random_file = random_file

    st.sidebar.info(f"Example file: {st.session_state.random_file}")
    image = load_image(f"Image/{st.session_state.random_file}")
    original_image_size_kb = None  # Not applicable for example images

if image is not None:
    st.sidebar.write(f"Image dimensions: {image.shape}")

    # Resize image for faster processing
    height, width = image.shape[:2]
    max_dim = 800
    if height > max_dim or width > max_dim:
        scale = max_dim / max(height, width)
        image = cv2.resize(image, (int(width * scale), int(height * scale)))

    # Split the image into its blue, green, and red channels (BGR format)
    blue, green, red = cv2.split(image)

    # Perform PCA on the blue channel to calculate explained variance
    pca_temp = PCA().fit(blue)
    explained_variance_ratio = pca_temp.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Function to generate a slider label with the number of components and variance information
    def slider_label(components):
        return f"Number of PCA Components: {components}, Variance Preserved: {cumulative_variance[components-1]:.2%}"

    # Create sliders for selecting the number of PCA components
    pca_components = st.slider(slider_label(20), 1, blue.shape[0], 20, format="%d")
    st.sidebar.write(f"Variance preserved: {cumulative_variance[pca_components-1]:.2%}")

    # Track performance
    start_time = time.time()
    st.spinner("Processing...")

    # Apply PCA to each channel
    redI = apply_pca(red, pca_components)
    greenI = apply_pca(green, pca_components)
    blueI = apply_pca(blue, pca_components)

    # Reconstruct the image
    re_image_bgr = (np.dstack((blueI, greenI, redI))).astype(np.uint8)
    re_image_rgb = cv2.cvtColor(re_image_bgr, cv2.COLOR_BGR2RGB)

    # Performance metrics
    elapsed_time = time.time() - start_time
    st.sidebar.write(f"Time taken: {elapsed_time:.2f} seconds")

    # Create two columns to display the original and compressed images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.header("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width="always")
    with col2:
        st.header(f"Compressed Image ({pca_components} Components, {cumulative_variance[pca_components-1]:.2%} Variance Preserved)")
        st.image(re_image_rgb, use_column_width="always")

    # Display sizes of original and compressed images
    if original_image_size_kb is not None:
        compressed_image_size_kb = len(cv2.imencode('.png', re_image_bgr)[1]) / 1024
        
        # Create containers for image sizes
        st.write("### Image Sizes")
        size_container = st.container()
        with size_container:
            st.write(f"**Original Image Size:** {original_image_size_kb:.2f} KB")
            st.write(f"**Compressed Image Size:** {compressed_image_size_kb:.2f} KB")

    # Provide an option to download the compressed image
    st.download_button(
        label="Download Compressed Image",
        data=cv2.imencode('.png', re_image_bgr)[1].tobytes(),
        file_name='compressed_image.png',
        mime='image/png'
    )
else:
    st.error("No image uploaded or selected. Please upload an image or select an example.")
         