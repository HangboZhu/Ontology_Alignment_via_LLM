import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, set_seed
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist


def compute_embeddings(texts, tokenizer, model, batch_size):
    all_reps = []
    for i in tqdm(np.arange(0, len(texts), batch_size), desc="Embedding"):
        toks = tokenizer.batch_encode_plus(
            texts[i:i+batch_size],
            padding="max_length",
            max_length=25,
            truncation=True,
            return_tensors="pt"
        )
        toks_cuda = {k: v.cuda() for k, v in toks.items()}
        output = model(**toks_cuda)
        cls_rep = output[0][:, 0, :]
        all_reps.append(cls_rep.cpu().detach().numpy())
        del toks, toks_cuda, output, cls_rep
        torch.cuda.empty_cache()
    return np.concatenate(all_reps, axis=0)


def compute_and_save_with_synonym(df, field, tokenizer, model, batch_size, synonym_df, out_path, mode):
    all_text = df[field].fillna("").tolist()
    print(f"\nEncoding field: {field}")
    all_emb = compute_embeddings(all_text, tokenizer, model, batch_size)
    query_emb = all_emb.copy()

    print("Computing similarity with synonym cutoff...")
    dist = cdist(query_emb, all_emb)
    sim = cosine_similarity(query_emb, all_emb)

    results = []

    for q_idx, q_text in tqdm(enumerate(all_text), total=len(all_text), desc="Matching with synonym"):
        dist_vec = dist[q_idx]
        sim_vec = sim[q_idx]
        query_id = df.iloc[q_idx]["id"]
        query_lbl = df.iloc[q_idx]["lbl"]
        query_description = df.iloc[q_idx]["meta_definition_val"]
        query_out = q_text if mode == 'lbl2lbl' else query_lbl

        syn_records = synonym_df[synonym_df["lbl"] == q_text] if mode == 'lbl2lbl' else synonym_df[synonym_df["meta_definition_val"] == query_description]

        matched_indices = []
        predicted_levels_map = {}
        synonym_mean_map = {}

        if syn_records.empty:
            threshold = 0.85
            default_level = "Raw data without synon, use 0.85"
            for idx, sim_score in enumerate(sim_vec):
                if idx != q_idx and sim_score > threshold:
                    matched_indices.append(idx)
                    predicted_levels_map[idx] = default_level
                    synonym_mean_map[idx] = threshold
        else:
            level_thresholds = syn_records.groupby("Synonym_Level")["cosine_similarity"].mean().to_dict()
            for idx, sim_score in enumerate(sim_vec):
                if idx == q_idx:
                    continue
                passed_levels = []
                for level, cutoff in level_thresholds.items():
                    if sim_score > cutoff:
                        passed_levels.append(level)
                if passed_levels:
                    matched_indices.append(idx)
                    predicted_levels_map[idx] = ",".join(passed_levels)
                    synonym_mean_map[idx] = level_thresholds.get(passed_levels[0], np.nan)

        if matched_indices:
            for idx in matched_indices:
                row = df.iloc[idx]
                results.append({
                    "query_id": query_id,
                    "query_lbl": query_out,
                    "query_description": query_description,
                    
                    "predicted_id": row["id"],
                    "predicted_label": row["lbl"],
                    "predicted_description": row["meta_definition_val"],
                    
                    "distance": dist_vec[idx],
                    "similarity": sim_vec[idx],
                    "predicted_synonym_level": predicted_levels_map.get(idx, ""),
                    "synonym_similarity_mean": synonym_mean_map.get(idx, ""),
                    "cluster": "has synonym"
                })
        else:
            results.append({
                "query_id": query_id,
                "query_lbl": query_out,
                "query_description": query_description,
                
                "predicted_id": "",
                "predicted_lbl": "",
                "predicted_description": "",
                
                "distance": "",
                "similarity": "",
                "predicted_synonym_level": "",
                "synonym_similarity_mean": "",
                "cluster": "self"
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_path, index=False)
    print(f"Saved result to: {out_path}")


import argparse
import os

def main():
    # === argparse 参数解析 ===
    parser = argparse.ArgumentParser(description="Run synonym matching with embeddings")
    parser.add_argument("-i", "--input", required=True, help="Input CSV file")
    parser.add_argument("-s", "--synonym", required=True, help="Synonym CSV file")
    parser.add_argument("--model", required=True, help="Path to pre-trained model")
    parser.add_argument("-o", "--output", required=True, help="Output folder")
    parser.add_argument("-b", "--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("-m", "--mode", choices=["lbl2lbl", "desc2desc", "all"], default="all", help="Mode of operation")
    args = parser.parse_args()

    # === 参数赋值 ===
    set_seed(33)
    data_path = args.input
    synonym_path = args.synonym
    model_path = args.model
    out_dir = args.output
    batch_size = args.batch_size
    mode = args.mode

    os.makedirs(out_dir, exist_ok=True)

    # === 加载资源 ===
    print("Loading data & model...")
    synonym_df = pd.read_csv(synonym_path)
    df_raw = pd.read_csv(data_path)
    # if mode == "lbl2lbl":
    #     df_raw = df_raw[df_raw["lbl"].notna()]
    # elif mode == "desc_2_desc":
    #     df_raw = df_raw[df_raw["meta_definition_val"].notna()]
    # elif mode == "all":
    #     df_raw = df_raw[df_raw["meta_definition_val"].notna() | df_raw["lbl"].notna()]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)  
    model = AutoModel.from_pretrained(model_path).cuda()

    # === 执行匹配逻辑 ===
    if mode in ["lbl2lbl", "desc2desc"]:
        field = "lbl" if mode == "lbl2lbl" else "meta_definition_val"
        # df = df_raw[df_raw[field].notna()].reset_index(drop=True)
        if mode == "lbl2lbl":
            df = df_raw[df_raw["lbl"].notna()].reset_index(drop=True)
        else:
            df = df_raw[df_raw["meta_definition_val"].notna()].reset_index(drop=True)
        compute_and_save_with_synonym(
            df=df,
            field=field,
            tokenizer=tokenizer,
            model=model,
            batch_size=batch_size,
            synonym_df=synonym_df,
            out_path=os.path.join(out_dir, f"{mode}_fusion_combined_results.csv"),
            mode=mode
        )
    elif mode == "all":
        for m, field in [("lbl2lbl", "lbl"), ("desc2desc", "meta_definition_val")]:
            if m == "lbl2lbl":
                df = df_raw[df_raw["lbl"].notna()].reset_index(drop=True)
            else:
                df = df_raw[df_raw["meta_definition_val"].notna()].reset_index(drop=True)
            compute_and_save_with_synonym(
                df=df,
                field=field,
                tokenizer=tokenizer,
                model=model,
                batch_size=batch_size,
                synonym_df=synonym_df,
                out_path=os.path.join(out_dir, f"{m}_fusion_combined_results.csv"),
                mode=m
            )

if __name__ == "__main__":
    main()

