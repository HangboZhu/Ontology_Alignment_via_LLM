import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
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
                                           return_tensors="pt")
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

        matched = [i for i, s in enumerate(sim_vec) if i != q_idx and s > cutoff]
        if matched:
            for idx in matched:
                row = df.iloc[idx]
                results.append({
                    "query": query_out,
                    "query_id": query_id,
                    "predicted_description": row["meta_definition_val"],
                    "predicted_id": row["id"],
                    "predicted_label": row["lbl"],
                    "distance": dist_vec[idx],
                    "similarity": sim_vec[idx],
                    "predicted_synonym_level": f"{field} cutoff > {cutoff}",
                    "synonym_similarity_mean": cutoff,
                    "cluster": "has synonym"
                })
        else:
            results.append({
                "query": query_out,
                "query_id": query_id,
                "predicted_description": "",
                "predicted_id": "",
                "predicted_label": "",
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
    parser.add_argument('-o', '--output', required=True, help='Output file or folder')
    parser.add_argument('-c', '--cutoff', type=float, default=0.85, help='Similarity cutoff')
    parser.add_argument('--model', required=True, help='HuggingFace model path')
    parser.add_argument('-b', '--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('-m', '--mode', required=True, choices=['lbl2lbl', 'desc2desc', 'all'], help='Comparison mode')

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    
    # 修改了这歌，是需要输入json转换完的csv文件就可以了 Json ==> flatten csv(把filter的代码整合到这里了)
    df = df[df["meta_deprecated"] != True]
    if args.mode == "lbl2lbl":
        df = df[df["lbl"].notna()]
    elif args.mode == "desc2desc":
        df = df[df["meta_definition_val"].notna()]
    else:
        df = df[df["lbl"].notna() & df["meta_definition_val"].notna()]
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).cuda()

    if args.mode == 'lbl2lbl':
        compute_and_save(df, 'lbl', tokenizer, model, args.batch_size, args.cutoff, args.output, args.mode)
    elif args.mode == 'desc2desc':
        compute_and_save(df, 'meta_definition_val', tokenizer, model, args.batch_size, args.cutoff, args.output, args.mode)
    elif args.mode == 'all':
        if not os.path.isdir(args.output):
            os.makedirs(args.output, exist_ok=True)
        compute_and_save(df, 'lbl', tokenizer, model, args.batch_size, args.cutoff,
                         os.path.join(args.output, 'lbl2lbl_result.csv'), args.mode)
        compute_and_save(df, 'meta_definition_val', tokenizer, model, args.batch_size, args.cutoff,
                         os.path.join(args.output, 'desc2desc_result.csv'), args.mode)
if __name__ == "__main__":
    main()
