from torchvision import transforms
from typing import Sequence
from PIL import Image
import hubconf
import logging
import colorlog
from tqdm import tqdm
import cv2
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import os

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Use timm's names for ImageNet default mean and standard deviation
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    """Create a composite transform for evaluation with classification models."""
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)


def setup_logger(level=logging.DEBUG):
    """
    Set up a logger with color support for debug level messages.

    Args:
        level: The logging level, defaults to logging.DEBUG.
    """
    logger = logging.getLogger("FeatureExtractionLogger")
    logger.setLevel(level)

    # Define log format with colors
    log_format = "%(log_color)s%(levelname)-8s%(reset)s - %(log_color)s%(message)s"
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(log_format))

    logger.addHandler(handler)

    return logger


def setup_transforms():
    """
    Setup the image transformations.
    """
    return transforms.Compose([
        transforms.Resize(256),  # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),  # Crop the center 224x224 pixels
        transforms.ToTensor(),  # Convert to PyTorch Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet mean and std
    ])



def create_output_dirs(video_path, save_raw_features, save_attentions, save_pca):
    """
    Create output directories for raw features, attentions, and PCA features.
    """
    base_dir = Path(video_path).parent / "dinov2_out"
    paths = {}
    if save_raw_features:
        paths['raw_features'] = base_dir / "raw_features"
    if save_attentions:
        paths['attentions'] = base_dir / "attentions"
    if save_pca:
        paths['pca_features'] = base_dir / "pca_features"

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths



def extract_features_from_frame(frame, model, transform):
    """
    Extract features from a single frame using the specified DINOv2 model and visualize
    the original and PCA-transformed image side by side.

    Args:
        frame: The video frame as a numpy array.
        model: The DINOv2 model for feature extraction.
        transform: The transformation to apply to the frame before feature extraction.
        logger: Logger for debug messages.
    """
    original_image = Image.fromarray(frame).convert("RGB")
    image = transform(original_image)
    image_tensor = image.unsqueeze(0).to(DEVICE)

    # Extract features
    with torch.no_grad():
        inference = model.forward_features(image_tensor)
        features = inference['x_norm_patchtokens'].detach().cpu().numpy()[0]

    logger.debug(f"Extracted features shape: {features.shape}")

    # if save_pca or save_attentions:
    #     #Apply PCA to reduce to 3 components (for RGB channels)
    #     pca = PCA(n_components=3)
    #     pca.fit(features)
    #     pca_features = pca.transform(features)
    #
    #
    #     pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    #     pca_features = pca_features * 255
    #     pca_image = pca_features.reshape(16, 16, 3).astype(np.uint8)
    #     pca_image_re= cv2.resize(pca_image, (width, height), interpolation=cv2.INTER_LINEAR)
    #     # Prepare subplot
    #     fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    #
    #     # Visualize the original image
    #     ax[0].imshow(original_image)
    #     ax[0].set_title("Original Image")
    #     ax[0].axis('off')
    #
    #     ax[1].imshow(pca_image_re)
    #     ax[1].set_title("PCA Image")
    #     ax[1].axis('off')
    #
    #     plt.show()

    return features

def extract_features_from_video(video_path, model, transform, logger, save_raw_features=True, save_attentions=True, save_pca=True, n_pca_components=3):
    """
    Extract features from each frame and save them along with optional PCA and attention images.
    """
    video_path = Path(video_path)
    output_dirs = create_output_dirs(video_path, save_raw_features, save_attentions, save_pca)

    vidcap = cv2.VideoCapture(str(video_path))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    success, frame = vidcap.read()
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    features_list = []
    frame_processed = 0

    with tqdm(total=min(frame_count, 100), desc="Extracting features", unit="frame") as pbar:
        while success and frame_processed < 100:
            # Convert the frame from BGR (OpenCV default) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Extract features from the RGB frame
            features = extract_features_from_frame(frame_rgb, model, transform,width, height, logger, save_attentions=True, save_pca=True)
            features_list.append(features)

            # Optional: Save attention images (adjust feature extraction to include attention maps)

            success, frame = vidcap.read()
            pbar.update(1)
            frame_processed += 1


    # Save raw features if requested
    if save_raw_features:
        np.save(str(output_dirs['raw_features'] / f"{video_path.stem}_features.npy"), features_list)

    logger.info(f"Processed {video_path.name}")


def frame_to_pca(frame_features,  output_shape, n_components=3,):
    """
    Apply PCA to the features of a single frame and resize to match the original video dimensions.
    """
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(frame_features)
    pca_features_normalized = np.clip(
        (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min()) * 255, 0, 255)
    pca_image = pca_features_normalized.reshape((16, 16, 3)).astype(np.uint8)

    # Resize the PCA image to match the original video dimensions
    pca_image_resized = cv2.resize(pca_image, output_shape, interpolation=cv2.INTER_LINEAR)
    return pca_image_resized


def video_to_pca_video(video_path, model, transform, output_video_path, fps=25):
    vidcap = cv2.VideoCapture(str(video_path))
    success, frame = vidcap.read()

    # Obtain original video dimensions
    original_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_shape = (original_width, original_height)

    # Initialize video writer with original video dimensions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, output_shape)

    while success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        features = extract_features_from_frame(frame_rgb, model, transform)

        # Resize the PCA-transformed frame to match the original video dimensions
        pca_frame = frame_to_pca(features, n_components=3, output_shape=output_shape)
        out.write(pca_frame)

        success, frame = vidcap.read()

    out.release()
    print(f"PCA video saved to {output_video_path}")

if __name__ == "__main__":
    video_path = "D:/GitHub/dino/data/NP001/infant.video.mp4"
    output_video_path = "D:/GitHub/dino/data/NP001/direct.pca.mp4"
    logger = setup_logger(level=logging.INFO)  # Setup colored logger
    model = hubconf.dinov2_vitl14().to(DEVICE)  # Load the DINOv2 model
    #transform = make_classification_eval_transform()  # Setup transformations
    transform = setup_transforms()

    # # Extract features from video
    # extract_features_from_video(video_path, model, transform, logger)
    #video_to_pca_video(video_path, model, transform, output_video_path)
    video_to_pca_video(video_path, model, transform, output_video_path)
