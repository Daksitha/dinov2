import argparse
import cv2
import logging
import numpy as np
import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from sklearn.decomposition import PCA
import hubconf
import colorlog
from tqdm import tqdm

def setup_logger(level=logging.DEBUG):
    logger = logging.getLogger("FeatureExtractionLogger")
    logger.setLevel(level)
    log_format = "%(log_color)s%(levelname)-8s%(reset)s - %(log_color)s%(message)s"
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(log_format))
    logger.addHandler(handler)
    return logger


def setup_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_features_from_frame(frame, model, transform, device):
    original_image = Image.fromarray(frame).convert("RGB")
    image = transform(original_image)
    image_tensor = image.unsqueeze(0).to(device)
    with torch.no_grad():
        inference = model.forward_features(image_tensor)
        features = inference['x_norm_patchtokens'].detach().cpu().numpy()[0]
    return features


def frame_feature_to_pca(frame_features, output_shape=None, n_components=3, return_pca_image=False):
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(frame_features)
    pca_image_resized = None
    if return_pca_image:
        pca_features_normalized = np.clip(
            ((pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())) * 255, 0, 255)
        pca_image = pca_features_normalized.reshape((16, 16, 3)).astype(np.uint8)
        pca_image_resized = cv2.resize(pca_image, output_shape, interpolation=cv2.INTER_LINEAR)
    return pca_features, pca_image_resized


def process_video(args, logger, device):
    model = hubconf.__dict__[f"dinov2_vit{args.model_type}14"]().to(device)
    transform = setup_transforms()
    vidcap = cv2.VideoCapture(args.video_path)
    success, frame = vidcap.read()
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_feature_list, pca_features_list = [], []

    # Initialize tqdm progress bar for the main processing loop
    with tqdm(total=total_frames, desc="Processing video frames") as pbar:
        while success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            features = extract_features_from_frame(frame_rgb, model, transform, device)
            pca_features, _ = frame_feature_to_pca(features, n_components=args.pca_components, return_pca_image=False)  # Video writing part removed

            if args.save_raw_features:
                raw_feature_list.append(features)
            if args.save_pca_features:
                pca_features_list.append(pca_features)

            success, frame = vidcap.read()
            pbar.update(1)

    vidcap.release()

    if args.save_raw_features:
        np.save(f"{Path(args.video_path)}.dinov2_vit{args.model_type}.stream.npy", np.array(raw_feature_list))
        logger.info(f"Saved raw features at {Path(args.video_path).stem}.dinov2_vit{args.model_type}.stream.npy")

    if args.save_pca_features:
        np.save(f"{Path(args.video_path)}.dinov2_vit{args.model_type}_{args.pca_components}pca.stream.npy",
                np.array(pca_features_list))
        logger.info(f"Saved PCA features at {Path(args.video_path).stem}.dinov2_vit{args.model_type}_{args.pca_components}pca.stream.npy")


def main():
    parser = argparse.ArgumentParser(description="Video feature extraction with DINOv2.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video.")
    parser.add_argument("--model_type", type=str, choices=["s", "m", "l", "g"], default="l",
                        help="Model type to use (s, m, l, g).")
    parser.add_argument("--save_raw_features", action="store_true", default=True, help="Save raw features.")
    parser.add_argument("--save_pca_features", action="store_true", default=True, help="Save PCA features.")
    parser.add_argument("--save_pca_video", action="store_true", default=False, help="Save PCA visualization video.")
    parser.add_argument("--pca_components", type=int, default=3, help="Number of PCA components.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(level=logging.INFO)
    process_video(args, logger, device)


if __name__ == "__main__":
    main()
