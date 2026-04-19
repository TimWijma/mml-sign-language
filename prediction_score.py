import pandas as pd
import sacrebleu
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util

PREDICTIONS_CSV = "results/predictions_mosaic_FINETUNED.csv"

df = pd.read_csv(PREDICTIONS_CSV)

references = df["sentence"].tolist()
predictions = df["predicted"].tolist()

# BLEU
bleu = sacrebleu.corpus_bleu(predictions, [references])

# BERTScore
P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)

# Semantic Similarity
sem_model = SentenceTransformer("all-MiniLM-L6-v2")
pred_embeddings = sem_model.encode(predictions, convert_to_tensor=True, show_progress_bar=True)
ref_embeddings = sem_model.encode(references, convert_to_tensor=True, show_progress_bar=True)

cosine_scores = util.cos_sim(pred_embeddings, ref_embeddings).diagonal()


print(f"BLEU: {bleu.score:.2f}")
print(f"BERTScore - Precision: {P.mean():.4f} | Recall: {R.mean():.4f} | F1: {F1.mean():.4f}")
print(f"Semantic Similarity (cosine): {cosine_scores.mean():.4f}")
