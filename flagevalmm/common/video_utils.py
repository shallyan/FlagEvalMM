from PIL import Image
import numpy as np
import decord
import av
import torch
from flagevalmm.common.logger import get_logger

logger = get_logger(__name__)


def read_video_pyav(video_path: str, max_num_frames: int, return_tensors: bool = False):
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    if total_frames > max_num_frames:
        indices = np.arange(0, total_frames, total_frames / max_num_frames).astype(int)
    else:
        indices = np.arange(total_frames)
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    if return_tensors:
        tensors = torch.from_numpy(frames).float().cuda()
        tensors = tensors.permute(0, 3, 1, 2)
        return tensors
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def load_image_or_video(
    image_or_video_path: str, max_num_frames: int, return_tensors: bool
):
    logger.info(f"Loading image or video from {image_or_video_path}")
    if image_or_video_path.endswith(".png"):
        frame = Image.open(image_or_video_path)
        frame = frame.convert("RGB")
        frame = np.array(frame).astype(np.uint8)
        frame_list = [frame]
        buffer = np.array(frame_list)
    elif image_or_video_path.endswith(".mp4"):
        decord.bridge.set_bridge("native")
        video_reader = decord.VideoReader(image_or_video_path, num_threads=1)

        frames = video_reader.get_batch(
            range(len(video_reader))
        )  # (T, H, W, C), torch.uint8
        buffer = frames.asnumpy().astype(np.uint8)
    else:
        buffer = None

    frames = buffer
    nums_frame = min(len(frames), max_num_frames)
    if nums_frame:
        frame_indices = np.linspace(
            start=0, stop=len(frames) - 1, num=nums_frame
        ).astype(int)
        frames = frames[frame_indices]

    if return_tensors:
        frames = torch.from_numpy(frames).float().cuda()
        frames = frames.permute(0, 3, 1, 2)

    return frames