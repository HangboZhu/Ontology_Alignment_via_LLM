import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers import set_seed
import torch
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
import os

def compute_embeddings(texts, tokenizer, model, batch_size):
    all_reps = []
    for i in tqdm(np.arange(0, len(texts), batch_size)):
        toks = tokenizer.batch_encode_plus(texts[i:i+batch_size],
                                           padding="max_length",
                                           max_length=25,
                                           truncation=True,
                                           return_tensors="pt") # why use this？ because it is a random seed for pretraining
        toks_cuda = {k: v.cuda() for k, v in toks.items()}
        output = model(**toks_cuda)
        cls_rep = output[0][:, 0, :]
        all_reps.append(cls_rep.cpu().detach().numpy())
        del toks, toks_cuda, output, cls_rep
        torch.cuda.empty_cache()
    return np.concatenate(all_reps, axis=0)

def compute_and_save(df, field, tokenizer, model, batch_size, cutoff, out_path, mode):
    all_text = df[field].fillna("").tolist()

    print(f"Encoding: {field}")
    all_emb = compute_embeddings(all_text, tokenizer, model, batch_size)
    query_emb = all_emb.copy()

    print("Computing similarity...")
    dist = cdist(query_emb, all_emb)
    sim = cosine_similarity(query_emb, all_emb)

    results = []

    for q_idx, q_text in tqdm(enumerate(all_text), total=len(all_text)):
        dist_vec = dist[q_idx]
        sim_vec = sim[q_idx]
        query_id = df.iloc[q_idx]["id"]
        query_lbl = df.iloc[q_idx]["lbl"]
        query_out = q_text if mode == 'lbl2lbl' else query_lbl
        query_description = df.iloc[q_idx]["meta_definition_val"]
        query_source = df.iloc[q_idx]["Source"]

        matched = [i for i, s in enumerate(sim_vec) if i != q_idx and s > cutoff]
        if matched:
            for idx in matched:
                row = df.iloc[idx]
                results.append({
                    "query_id": query_id,
                    "query_lbl": query_out,
                    "query_description": query_description, 
                    "query_source": query_source,
                    
                    "predicted_id": row["id"],
                    "predicted_lbl": row["lbl"],
                    "predicted_description": row["meta_definition_val"],
                    "predicted_source": row["Source"],
                    
                    "distance": dist_vec[idx],
                    "similarity": sim_vec[idx],
                    "predicted_synonym_level": f"{field} cutoff > {cutoff}",
                    "synonym_similarity_mean": cutoff,
                    "cluster": "has synonym"
                })
        else:
            results.append({
                    "query_id": query_id,
                    "query_lbl": query_out,
                    "query_description": query_description, 
                    "query_source": query_source,
                    
                    "predicted_id": "",
                    "predicted_lbl": "",
                    "predicted_description": "",
                    "predicted_source": "",
                    
                    "distance": "",
                    "similarity": "",
                    "predicted_synonym_level": "",
                    "synonym_similarity_mean": "",
                    "cluster": "self"
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_path, index=False)
    print(f"Saved result to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Similarity matching using embeddings")
    parser.add_argument('-i', '--input', required=True, help='Input CSV file')
    parser.add_argument('-o', '--output', required=True, help='Output folder')
    parser.add_argument('-c', '--cutoff', type=float, default=0.85, help='Similarity cutoff')
    parser.add_argument('--model', required=True, help='HuggingFace model path')
    parser.add_argument('-b', '--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('-m', '--mode', required=True, choices=['lbl2lbl', 'desc2desc', 'all'], help='Comparison mode')

    args = parser.parse_args()
    set_seed(33)  # Why this because it is a random seed for pretraining

    df = pd.read_csv(args.input)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).cuda()

    if args.mode == 'lbl2lbl':
        print("Processing label to label similarity...")
        df = df[df["lbl"].notna()]
        compute_and_save(df, 'lbl', tokenizer, model, args.batch_size, args.cutoff, os.path.join(args.output, "lbl2lbl_result.csv"), args.mode)
    elif args.mode == 'desc2desc':
        print("Processing description to description similarity...")
        df = df[df["meta_definition_val"].notna()]
        compute_and_save(df, 'meta_definition_val', tokenizer, model, args.batch_size, args.cutoff, os.path.join(args.output, "desc2desc_result.csv"), args.mode)
    elif args.mode == 'all':
        print("Processing both label and description similarities...")
        # if not os.path.isdir(args.output):
        #     os.makedirs(args.output, exist_ok=True)
        # print("Processing label to label similarity...")
        # compute_and_save(df, 'lbl', tokenizer, model, args.batch_size, args.cutoff,
        #                  os.path.join(args.output, 'lbl2lbl_result.csv'), args.mode)
        # print("Processing description to description similarity...")
        # compute_and_save(df, 'meta_definition_val', tokenizer, model, args.batch_size, args.cutoff,
        #                  os.path.join(args.output, 'desc2desc_result.csv'), args.mode)
        # for field in ['lbl', 'desc2desc']:
        #     print(f"Processing {field} similarity...")
        #     if field == 'lbl':
        #         df = df[df["lbl"].notna()]
        #     elif field == 'desc2desc':
        #         df = df[df["meta_definition_val"].notna()]
        #     compute_and_save(df, field, tokenizer, model, args.batch_size, args.cutoff,
        #                      os.path.join(args.output, f"{field}_result.csv"), args.mode)
        for field, mode_field in [('lbl', 'lbl2lbl'), ('meta_definition_val', 'desc2desc')]:
            print(f"Processing {mode_field} similarity...")
            df_field = df[df[field].notna()]
            compute_and_save(df_field, field, tokenizer, model, args.batch_size, args.cutoff,
                            os.path.join(args.output, f"{mode_field}_result.csv"), args.mode)

if __name__ == "__main__":
    main()
