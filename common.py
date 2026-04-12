import io

import torch
from PIL import Image

def convert_row(row, original_fps=30, target_fps=24, resolution=(512, 360), include_description=True, mode="normal"):
	description = row["sentence"] if include_description else ""

	if mode == "mosaic":
		gif_bytes = row.get("mosaic_gif") or row.get("video")
		gif = Image.open(io.BytesIO(gif_bytes))
		frames = []
		for i in range(gif.n_frames):
			gif.seek(i)
			frames.append(gif.convert("RGB"))
	else:
		decoder = row["video"]
		total_frames = decoder.metadata.num_frames
		duration_sec = total_frames / original_fps
		target_fps = float(target_fps)
		n_frames = max(4, int(target_fps * duration_sec))
		indices = torch.linspace(0, total_frames - 1, n_frames).long().tolist()
		frames = decoder.get_frames_at(indices).data  # (T, C, H, W)
		frames = frames.permute(0, 2, 3, 1)           # → (T, H, W, C)
		frames = frames[:, :, 128:1152, :]            # Crop
		frames = [Image.fromarray(f.numpy()) for f in frames]

	messages = [
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
				{
					"type": "video", 
					"video": frames, 
					"max_pixels": resolution[0] * resolution[1],
				},
				{
					"type": "text", 
					"text": "Translate the sign language in this video into English. Carefully watch the signer's hand movements, and facial expressions."
				},
			],
		},
	]

	if include_description:
		messages.append({
			"role": "assistant",
			"content": [
				{
					"type": "text",
					"text": description
				}
			]
		})

	return messages, frames