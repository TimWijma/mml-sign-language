import os
import torch
import pandas as pd

from unsloth import FastVisionModel

from datasets import load_dataset
from common import convert_row

MAX_SEQ_LENGTH = 20000
TARGET_FPS = 24
RESOLUTION = (512, 360)

MODE = "normal"	# "normal" or "mosaic"

DATASET_NAME = "Flimdejong/how2sign-3s" if MODE == "normal" else "TimWijma/how2sign-3s-mosaic"

MODEL_NAME = "final_loras/crop_center"
RESULTS_PATH = "results/predictions_crop_center_FINETUNED.csv"

model, tokenizer = FastVisionModel.from_pretrained(
	MODEL_NAME,
	load_in_4bit=True
)
FastVisionModel.for_inference(model)

dataset = load_dataset(DATASET_NAME, split="test")

print(f"Dataset {DATASET_NAME} loaded:\n {dataset}")

def predict_batch(dataset, batch_size=4, output_path="results/predictions.csv"):
	results = []
	rows = list(dataset)
	
	for i in range(0, len(rows), batch_size):
		batch = rows[i:i+batch_size]

		batch_texts = []
		batch_frames = []

		for row in batch:
			messages, frames = convert_row(row, target_fps=TARGET_FPS, resolution=RESOLUTION, include_description=False, mode=MODE)
			text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
			batch_texts.append(text)
			batch_frames.append(frames)

		inputs = tokenizer(
			text=batch_texts,
			videos=batch_frames,
			return_tensors="pt",
			padding=True,
			truncation=True,
		).to("cuda")

		with torch.no_grad():
			output_ids = model.generate(
				**inputs,
				max_new_tokens=128,
				use_cache=True,
				do_sample=False,
				repetition_penalty=1.0,
				eos_token_id=tokenizer.eos_token_id,
				pad_token_id=tokenizer.pad_token_id,
			)

		for j, (out, inp_ids) in enumerate(zip(output_ids, inputs["input_ids"])):
			input_len = inp_ids.shape[0]
			generated = tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
			row = batch[j]
			results.append({
				"clip_id": row["clip_id"],
				"sentence": row["sentence"],
				"predicted": generated,
			})

		print(f"Processed {min(i+batch_size, len(rows))}/{len(rows)}")

	os.makedirs(os.path.dirname(output_path), exist_ok=True)

	df = pd.DataFrame(results)
	df.to_csv(output_path, index=False)
	print(f"Saved {len(df)} predictions to {output_path}")
	return df

df = predict_batch(dataset, batch_size=8, output_path=RESULTS_PATH)