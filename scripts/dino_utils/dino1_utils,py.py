def generate_video_from_attention_maps(input_dir, output_path, video_format='mp4', fps=30):
    """
    Generate a video from attention maps stored as images in a directory.

    Args:
        input_dir (str): Directory containing the attention map images.
        output_path (str): Path (including filename and extension) where the video will be saved.
        video_format (str): Format of the output video ('mp4' or 'avi').
        fps (int): Frames per second for the output video.
    """
    img_array = []
    attention_images_list = sorted(glob.glob(os.path.join(input_dir, "attn-*.jpg")))

    if not attention_images_list:
        print("No attention maps found in the specified directory.")
        return

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"MP4V") if video_format == 'mp4' else cv2.VideoWriter_fourcc(*"XVID")

    # Get size of the first image
    with open(attention_images_list[0], "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")
        size = (img.width, img.height)

    print(f"Generating video {size} to {output_path}")

    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    for filename in tqdm(attention_images_list):
        with open(filename, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            img_array.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

    for i in img_array:
        out.write(i)
    out.release()
    print("Done")



@torch.no_grad()
def extract_features_and_save_attention_maps(model, data_loader, output_dir, use_cuda=True, save_attn_maps=True,
                                             patch_size=16, multiscale=False, threshold=0.6):
    """
    Extract features and optionally save attention maps.

    Args:
        model: The Vision Transformer model.
        data_loader: DataLoader for the dataset.
        output_dir: Directory to save attention maps.
        use_cuda: Whether to use CUDA.
        save_attn_maps: Whether to save attention maps.
        patch_size: The patch size used by the Vision Transformer model.
        multiscale: inference from multiple scales of the input images to capture a more comprehensive
                    representation of the features.
        threshold:  visualize masks obtained by thresholding the self-attention maps to keep xx percent of the mass.
    """
    if use_cuda:
        model.cuda()

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = []
    for batch_idx, (samples, indices, paths) in enumerate(data_loader, start=1):
        if use_cuda:
            samples = samples.cuda(non_blocking=True)

        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            # feats size : batch size * 364 for batch size * 3*224*224 images
            feats = model(samples).clone()

        features.append(feats.cpu())

        if save_attn_maps:
            # Assuming get_last_selfattention outputs attention maps for the batch
            #attn_maps = model.get_last_selfattention(samples)
            for i, img in enumerate(samples):
                w, h = (
                    img.shape[1] - img.shape[1] % patch_size,
                    img.shape[2] - img.shape[2] % patch_size,
                )
                img = img[:, :w, :h].unsqueeze(0)

                w_featmap = img.shape[-2] // patch_size
                h_featmap = img.shape[-1] // patch_size

                attentions = model.get_last_selfattention(img.to(DEVICE))
                # attention data of size: 1,6,785,785 for image h:224,2:224
                nh = attentions.shape[1]  # number of head

                # we keep only the output patch attention
                # not it is only 6,1585
                attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

                # we keep only a certain percentage of the mass
                val, idx = torch.sort(attentions)
                val /= torch.sum(val, dim=1, keepdim=True)
                cumval = torch.cumsum(val, dim=1)
                th_attn = cumval > (1 - threshold)
                idx2 = torch.argsort(idx)
                for head in range(nh):
                    th_attn[head] = th_attn[head][idx2[head]]
                th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
                # interpolate
                th_attn = (
                    nn.functional.interpolate(
                        th_attn.unsqueeze(0),
                        scale_factor=patch_size,
                        mode="nearest",
                    )[0]
                    .cpu()
                    .numpy()
                )

                attentions = attentions.reshape(nh, w_featmap, h_featmap)
                attentions = (
                    nn.functional.interpolate(
                        attentions.unsqueeze(0),
                        scale_factor=patch_size,
                        mode="nearest",
                    )[0]
                    .cpu()
                    .numpy()
                )

                # save attentions heatmaps
                fname = os.path.join(output_dir, "attn-" + os.path.basename(paths[i]))
                plt.imsave(
                    fname=fname,
                    arr=sum(
                        attentions[i] * 1 / attentions.shape[0]
                        for i in range(attentions.shape[0])
                    ),
                    cmap="inferno",
                    format="jpg",
                )

            # for i, attn_map in enumerate(attn_maps):
            #     # Here we assume the attention map is averaged over all heads for simplicity
            #     #attn_map = attn_map.mean(dim=1)  # Average over heads
            #     img_name = os.path.basename(paths[i])
            #     img_name = os.path.splitext(img_name)[0]
            #
            #     save_attention_maps(attn_map[i], output_dir, img_name, samples[i].shape[-2:], patch_size)

    features = torch.cat(features, dim=0)
    return features


@torch.no_grad()
def extract_attention_maps_and_save_as_video(model, data_loader, out_dir, role, use_cuda=True, patch_size=8,
                                             threshold=0.6, fps=25, video_format='mp4'):
    """
    Generate attention maps for each image in the dataset and save directly as a video.

    Args:
        model: Vision Transformer model capable of returning attention maps.
        data_loader: DataLoader providing batches of images.
        output_video_path: Path to save the output video.
        use_cuda: If True, use CUDA for model and data.
        patch_size: Size of the patches used by the Vision Transformer model.
        threshold: Threshold to apply on attention maps for visualization.
        fps: Frames per second for the output video.
        video_format: Format of the output video ('mp4' or 'avi').
    """
    if use_cuda:
        model.cuda()

    model.eval()
    # Initialize video writer
    video_writer = None
    output_video_path = Path(out_dir/f"{role}_attentions.mp4")

    for batch_idx, (samples, _, _,_) in tqdm(enumerate(data_loader), total=len(data_loader)):
        if use_cuda:
            samples = samples.cuda(non_blocking=True)

        for i, img in enumerate(samples):
            w, h = (
                img.shape[1] - img.shape[1] % patch_size,
                img.shape[2] - img.shape[2] % patch_size,
            )
            img = img[:, :w, :h].unsqueeze(0)

            w_featmap = img.shape[-2] // patch_size
            h_featmap = img.shape[-1] // patch_size

            attentions = model.get_last_selfattention(img.to(DEVICE))
            nh = attentions.shape[1]  # number of head

            # we keep only the output patch attention
            # not it is only 6,1585
            attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            th_attn = (
                nn.functional.interpolate(
                    th_attn.unsqueeze(0),
                    scale_factor=patch_size,
                    mode="nearest",
                )[0]
                .cpu()
                .numpy()
            )

            attentions = attentions.reshape(nh, w_featmap, h_featmap)
            attentions = (
                nn.functional.interpolate(
                    attentions.unsqueeze(0),
                    scale_factor=patch_size,
                    mode="nearest",
                )[0]
                .cpu()
                .numpy()
            )

            # Calculate the attention map (simplified for explanation)
            # attention_map = sum(
            #         attentions[i] * 1 / attentions.shape[0]
            #         for i in range(attentions.shape[0])
            #     )

            # Normalize and apply colormap
            norm = Normalize(vmin=attentions.min(), vmax=attentions.max())
            attention_map_normalized = plt.cm.inferno(norm(attentions))

            # Convert to BGR format for OpenCV and remove alpha channel
            attention_map_bgr = (attention_map_normalized[..., :3] * 255).astype(np.uint8)
            attention_map_bgr = cv2.cvtColor(attention_map_bgr, cv2.COLOR_RGB2BGR)

            if video_writer is None:
                h, w = attention_map_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

            # Convert processed attention map to BGR for video writing
            #attention_map_bgr = cv2.cvtColor(attentions.cpu().numpy(), cv2.COLOR_GRAY2BGR)
            video_writer.write(attention_map_bgr)

    video_writer.release()
    print(f"Finished saving attention maps as video:write to {output_video_path}.")


@torch.no_grad()
def save_attention_maps(model, data_loader, output_dir, use_cuda=True, patch_size=16, threshold=0.6):
    """
    Generate and save attention maps for each image in the dataset.

    Args:
        model: Vision Transformer model with a method `get_last_selfattention` to get attention maps.
        data_loader: DataLoader providing batches of images along with their paths.
        output_dir: Directory where attention maps will be saved.
        use_cuda: If True, use CUDA for model and data.
        patch_size: Size of the patches used by the Vision Transformer model.
        threshold: Threshold to apply on attention maps for visualization.
    """
    if use_cuda:
        model = model.cuda()

    model.eval()
    attention_maps_dir = Path(output_dir) / "attention_maps"
    attention_maps_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, (samples, indices, paths, original_sizes) in enumerate(data_loader, start=1):

        for i, img in enumerate(samples):
            w, h = (
                img.shape[1] - img.shape[1] % patch_size,
                img.shape[2] - img.shape[2] % patch_size,
            )
            img = img[:, :w, :h].unsqueeze(0)

            w_featmap = img.shape[-2] // patch_size
            h_featmap = img.shape[-1] // patch_size

            attentions = model.get_last_selfattention(img.to(DEVICE))
            # attention data of size: 1,6,785,785 for image h:224,2:224
            nh = attentions.shape[1]  # number of head

            # we keep only the output patch attention
            # not it is only 6,1585
            attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            th_attn = (
                nn.functional.interpolate(
                    th_attn.unsqueeze(0),
                    scale_factor=patch_size,
                    mode="nearest",
                )[0]
                .cpu()
                .numpy()
            )

            attentions = attentions.reshape(nh, w_featmap, h_featmap)
            attentions = (
                nn.functional.interpolate(
                    attentions.unsqueeze(0),
                    scale_factor=patch_size,
                    mode="nearest",
                )[0]
                .cpu()
                .numpy()
            )

            # save attentions heatmaps
            fname = os.path.join(output_dir, "attn-" + os.path.basename(paths[i]))
            plt.imsave(
                fname=fname,
                arr=sum(
                    attentions[i] * 1 / attentions.shape[0]
                    for i in range(attentions.shape[0])
                ),
                cmap="inferno",
                format="jpg",
            )

    print("Finished saving attention maps.")
