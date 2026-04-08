import torch
from PIL import Image
from torch.utils.data import Dataset

def convert_row(row, target_fps=8, resolution=(512, 360)):
    decoder = row["video"]
    description = row["sentence"]
    original_fps = 30
    total_frames = decoder.metadata.num_frames
    duration_sec = total_frames / original_fps

    n_frames = max(4, int(target_fps * duration_sec))
    indices = torch.linspace(0, total_frames - 1, n_frames).long().tolist()
    frames = decoder.get_frames_at(indices).data  # (T, C, H, W)
    frames = frames.permute(0, 2, 3, 1)           # → (T, H, W, C)
    frames = frames[:, :, 128:1152, :]            # Crop teeny tiny bit, original was 1280 x 720
    frames = [Image.fromarray(f.numpy()) for f in frames]
    w, h = resolution

    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a sign language interpreter. Translate signs accurately into English."
            },
            
            {
                "role": "user",
                "content": [
                    {
                    "type": "video",
                        "video": frames,
                        "max_pixels": w * h,
                    },
                    {"type": "text", "text": "Translate the sign language in this video into English."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": description}],
            },
        ]
    }

class ASLVideoDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return convert_row(self.data[idx])