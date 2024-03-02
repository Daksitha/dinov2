import glob
import cv2
from torchvision import transforms as pth_transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
#import vision_transformer as vits
#import utils
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
from pathlib import Path
from PIL import Image
import cv2
from torch.utils.data import Dataset
import concurrent.futures
import os
from pathlib import Path
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VideoDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [os.path.join(directory, img) for img in os.listdir(directory) if img.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        original_size = image.size
        # Extract frame number from the filename assuming format 'frame_XXXX.jpg'
        frame_number = int(os.path.basename(img_path).split('_')[-1].split('.')[0])

        if self.transform:
            image = self.transform(image)

        return image, idx, img_path, original_size, frame_number

def extract_frames_from_video(video_path, output_dir):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    frames = []
    while success:
        frame_filename = os.path.join(output_dir, f"frame_{count:04d}.jpg")
        frames.append((frame_filename, image))
        success, image = vidcap.read()
        count += 1
        if len(frames) >= 100:
            write_frames_to_disk(frames)
            frames = []
    if frames:
        # remaining frames
        write_frames_to_disk(frames)

def write_frames_to_disk(frames):
    for frame_filename, image in frames:
        cv2.imwrite(frame_filename, image)

def prepare_role_datasets_parallel(base_dir, role, output_base_dir):
    session_dirs = list(Path(base_dir).glob(f"*/{role}.video.mp4"))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for video_path in session_dirs:
            session_id = video_path.parent.stem
            output_dir_structure = {
                "frames": os.path.join(output_base_dir, session_id, role, "frames"),
                "features": os.path.join(output_base_dir, session_id, role, "extractions"),
                "attention_maps": os.path.join(output_base_dir, session_id, role, "attention_maps")
            }

            for dir_path in output_dir_structure.values():
                os.makedirs(dir_path, exist_ok=True)

            futures.append(executor.submit(extract_frames_from_video, str(video_path), output_dir_structure["frames"]))

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    """
    Extract features from images using the provided model.

    Args:
        model: The Vision Transformer model.
        data_loader: DataLoader for the dataset.
        use_cuda: Whether to use CUDA.
        multiscale: Whether to infer from multiple scales of the input images.

    Returns:
        A tensor containing extracted features for all images in the dataset.
    """
    if use_cuda:
        model.cuda()

    model.eval()
    features = []

    for batch_idx, (samples, indices, paths, original_sizes) in enumerate(data_loader, start=1):
        if use_cuda:
            samples = samples.cuda(non_blocking=True)

        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples).clone()

        features.append(feats.cpu())

    return torch.cat(features, dim=0)

def load_model(arch, patch_size, pretrained_weights, checkpoint_key=None):
    # Build model
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(DEVICE)

    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Taking key {checkpoint_key} from provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}  # Adjust if needed
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Pretrained weights found at {pretrained_weights} and loaded with msg: {msg}")
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Loading reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("No reference weights available for this model configuration. Using random weights.")

    return model

def main(base_dir, role, output_base_dir, model, transform, batch_size=64, patch_size=8):
    model, autocast_dtype = setup_and_build_model(args)
    prepare_role_datasets(base_dir, role, output_base_dir)
    session_dirs = list(Path(output_base_dir).glob(f"*/{role}/frames"))

    for session_dir in session_dirs:
        dataset = CustomDataset(str(session_dir), transform=transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        #save_attention_maps(model, data_loader,output_dir=session_dir.parent, use_cuda=DEVICE.type == 'cuda', patch_size=patch_size)
        extract_attention_maps_and_save_as_video(model, data_loader, session_dir.parent,role, use_cuda=True, patch_size=8,
                                             threshold=0.6, fps=25, video_format='mp4')
        #session_features = extract_features_and_save_attention_maps(model, data_loader,output_dir=session_dir.parent/"attention_maps", use_cuda=DEVICE.type == 'cuda', patch_size=patch_size)
        #features_path = os.path.join(session_dir.parent, "extractions", "features.pt")
        #torch.save(session_features, features_path)


if __name__ == "__main__":
    # Example usage
    base_dir = "data"
    role = "infant"
    output_base_dir = "data/output"

    arch = "vit_small"
    patch_size = 8
    pretrained_weights = "models/dino_deitsmall8_pretrain.pth"
    checkpoint_key = None
    resize = None

    if resize is not None:
        transform = pth_transforms.Compose(
            [
                pth_transforms.ToTensor(),
                pth_transforms.Resize(resize),
                pth_transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )
    else:
        transform = pth_transforms.Compose(
            [
                pth_transforms.ToTensor(),
                pth_transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

    # Initialize your model here
    model = load_model(arch, patch_size, pretrained_weights, checkpoint_key)

    main(base_dir, role, output_base_dir, model, transform,patch_size=patch_size)
