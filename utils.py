import streamlit as st
import cv2
import os
import numpy as np  # Missing import for numpy
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD

# Cache the directory listing to avoid repeated I/O operations
@st.cache_data
def get_image_list(directory="Image"):
    return os.listdir(directory)

# Cache the image reading to avoid repeated decoding
@st.cache_data
def load_image(file_path):
    return cv2.imread(file_path)

# Function to perform PCA and inverse transform
def apply_pca(channel, components):
    try:
        pca = PCA(n_components=components)
        transformed = pca.fit_transform(channel)
        return pca.inverse_transform(transformed)
    except Exception as e:
        st.error(f"Error applying PCA: {e}")
        return channel  # Return the original channel if PCA fails

def apply_kernel_pca(channel, components, kernel="rbf"):
    try:
        kpca = KernelPCA(n_components=components, kernel=kernel, fit_inverse_transform=True)
        transformed = kpca.fit_transform(channel)
        return kpca.inverse_transform(transformed)
    except Exception as e:
        st.error(f"Error applying Kernel PCA: {e}")
        return channel  # Return the original channel if Kernel PCA fails

def apply_truncated_svd(channel, components):
    try:
        svd = TruncatedSVD(n_components=components)
        transformed = svd.fit_transform(channel)
        return svd.inverse_transform(transformed)
    except Exception as e:
        st.error(f"Error applying Truncated SVD: {e}")
        return channel  # Return the original channel if SVD fails

# Function to compress image using a target file size
def compress_image_to_target_size(red, green, blue, target_size_kb, method='PCA', kernel=None):
    pca_components = 50
    jpeg_quality = 95
    tolerance_kb = 5  # Allowable difference from the target size

    while True:
        # Apply the chosen dimension reduction method
        if method == 'PCA':
            redI = apply_pca(red, pca_components)
            greenI = apply_pca(green, pca_components)
            blueI = apply_pca(blue, pca_components)
        elif method == 'Kernel PCA':
            redI = apply_kernel_pca(red, pca_components, kernel)
            greenI = apply_kernel_pca(green, pca_components, kernel)
            blueI = apply_kernel_pca(blue, pca_components, kernel)
        elif method == 'Truncated SVD':
            redI = apply_truncated_svd(red, pca_components)
            greenI = apply_truncated_svd(green, pca_components)
            blueI = apply_truncated_svd(blue, pca_components)

        # Reconstruct the image
        re_image_bgr = (np.dstack((blueI, greenI, redI))).astype(np.uint8)

        # Save image as JPEG with specified quality
        _, buffer = cv2.imencode('.jpg', re_image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        compressed_size_kb = len(buffer) / 1024  # Get compressed size in KB

        # Check if the file size is close enough to the target
        if abs(compressed_size_kb - target_size_kb) < tolerance_kb:
            break
        elif compressed_size_kb > target_size_kb:
            # If too large, reduce PCA components or decrease JPEG quality
            if pca_components > 5:
                pca_components -= 5
            else:
                jpeg_quality -= 5
        else:
            # If too small, increase PCA components or increase JPEG quality
            if pca_components < red.shape[0]:
                pca_components += 5
            else:
                jpeg_quality += 5

    return re_image_bgr, pca_components, jpeg_quality, compressed_size_kb
