import torch
import torch.nn.functional as F
import torchvision.transforms as T
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

# Function to load the appropriate model based on feature dimension
def load_model(feat_dim):
    model_map = {
        384: 'dinov2_vits14',
        768: 'dinov2_vitb14',
        1024: 'dinov2_vitl14',
        1536: 'dinov2_vitg14'
    }
    model_name = model_map.get(feat_dim, 'dinov2_vitg14')  # Default to vitg14 if dim not found
    model = torch.hub.load('facebookresearch/dinov2', model_name).to(device)
    return model

# Function to perform PCA and return transformed features
def perform_pca(features, n_components=3):
    pca = PCA(n_components=n_components)
    pca.fit(features)
    pca_features = pca.transform(features)
    return pca, pca_features

# Function to plot histograms of PCA components
def plot_pca_histograms(pca_features):
    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.hist(pca_features[:, i])
    plt.show()

# Function to plot original and PCA images
def plot_original_and_pca_images(orig_images, pca_images, nrows=2, ncols=2):
    fig, axes = plt.subplots(nrows, ncols*2, figsize=(12, 6))
    for i, (orig, pca) in enumerate(zip(orig_images, pca_images)):
        row = i // ncols
        col = (i % ncols) * 2
        axes[row, col].imshow(orig)
        axes[row, col].axis('off')
        axes[row, col+1].imshow(pca)
        axes[row, col+1].axis('off')
    plt.tight_layout()
    plt.show()

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

patch_h = 37
patch_w = 37
feat_dim = 1536  # Change this to 384, 768, 1024, or 1536 to use different models

# Transform for input images
transform = T.Compose([T.Resize(patch_h*14, interpolation=T.InterpolationMode.LANCZOS),
                       T.CenterCrop(patch_h*14),
                       T.ToTensor(),
                       T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

# Load the model
dinov2_model = load_model(feat_dim)

# Process images
imgs_tensor = torch.zeros(4, 3, patch_h * 14, patch_w * 14)
orig_images = []
for i in range(4):
    img_path = f"S:/FER/data/{i}/0000{i}00.png"
    img = Image.open(img_path).convert('RGB')
    orig_images.append(img)
    imgs_tensor[i] = transform(img)[:3]

with torch.no_grad():
    features_dict = dinov2_model.forward_features(imgs_tensor.to(device))
    features = features_dict['x_norm_patchtokens'].detach().cpu().numpy().reshape(-1, feat_dim)

# Perform PCA
_, pca_features = perform_pca(features)

# Segment using the first component
pca_features_bg = pca_features[:, 0] < 10
pca_features_fg = ~pca_features_bg

# PCA for only foreground patches
pca, pca_features_fg_transformed = perform_pca(features[pca_features_fg])

# Prepare PCA images for plotting
pca_images = np.zeros_like(pca_features).reshape(4, patch_h, patch_w, -1)
pca_images[..., 0] = pca_features_bg.reshape(4, patch_h, patch_w)
pca_images[..., 1] = pca_features_fg.reshape(4, patch_h, patch_w)

# Plot original images next to PCA images
plot_original_and_pca_images(orig_images, pca_images)

# Note: You may need to adjust the segmentation and PCA image preparation logic to match your specific requirements.
