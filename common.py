import torch
from PIL import Image

def convert_row(row):
    decoder = row["video"]
    description = row["sentence"]
    original_fps = 30
    total_frames = decoder.metadata.num_frames
    duration_sec = total_frames / original_fps

    # Target ~8 fps
    target_fps = float(10.0)
    n_frames = max(4, int(target_fps * duration_sec))
    
    indices = torch.linspace(0, total_frames - 1, n_frames).long().tolist()
    frames = decoder.get_frames_at(indices).data  # (T, C, H, W)
    frames = frames.permute(0, 2, 3, 1)           # → (T, H, W, C)
    frames = frames[:, :, 128:1152, :]            # Crop
    frames = [Image.fromarray(f.numpy()) for f in frames]

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert American Sign Language (ASL) interpreter. "
                    "Translate ASL from video into fluent, natural English. "
                    "Focus on hand shapes, movements, and facial expressions. "
                    "If a sign is ambiguous, provide your best interpretation. "
                    "Output only the English translation, nothing else."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames_np, "max_pixels": 512 * 360 * n_frames},
                    {"type": "text", "text": "Translate the sign language in this video into English. Carefully watch the signer's hand movements, and facial expressions."},
                ],
            },
        ]
    }