import torch
from PIL import Image
from torch.utils.data import Dataset

def convert_row(row, target_fps=4, resolution=(336, 192)):
    decoder = row["video"]
    description = row["sentence"]
    total_frames = decoder.metadata.num_frames
    duration_sec = total_frames / 30

    n_frames = int(target_fps * duration_sec)
    n_frames = max(16, min(n_frames, 64))

    indices = torch.linspace(0, total_frames - 1, n_frames).long().tolist()
    frames = decoder.get_frames_at(indices).data
    frames = frames.permute(0, 2, 3, 1)
    frames = [Image.fromarray(f.numpy()).resize(resolution) for f in frames]

    # Build one image content block per frame instead of a video block
    image_content = [
        {"type": "image", "image": frame}
        for frame in frames
    ]

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe what the woman is signing in ASL"},
                    *image_content
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