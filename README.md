# CQAI_PCA

# Image Compression using PCA

## Overview

This project demonstrates image compression using Principal Component Analysis (PCA) for dimensionality reduction. By reducing the dimensionality of image data, we can compress images effectively while retaining most of the important features, thus optimizing storage space without significantly compromising image quality.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
  
## Introduction

Image compression is crucial for efficient storage and transmission of images in various applications. PCA is a statistical technique that transforms data into a set of orthogonal components, capturing the most significant variance in the data. This project applies PCA to compress images by reducing their dimensionality.

## Features

- Compress grayscale and color images using PCA
- Adjustable number of principal components for compression level control
- Visual comparison between original and reconstructed images

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- scikit-learn
- Pillow (PIL)
- opencv-python
- setuptools
 
  

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git

2. **Navigate to the project directory:**
   ```bash
   cd your-repo-name

3. **Install the Required packages:**
   ```bash
   pip install -r requirements.txt


## Usage


1. Prepare your images:
Upload the images you want to compress using the file uploader  or you can also use input_images/ directory for trial purposes. Supported formats include JPEG, PNG, etc.

2. Run the Script:
   ```bash
   python pca_v2.py

## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name

3. commit your changes:
   ```bash
   git commit -m 'Add new feature'

4. Push to the branch:
   ```bash
   git push origin feature-name

5. Open a Pull Request.


### The app is live and you can access it using the source link below:
[Source](https://image-comp-3e0a69cbd4f9.herokuapp.com/
)
