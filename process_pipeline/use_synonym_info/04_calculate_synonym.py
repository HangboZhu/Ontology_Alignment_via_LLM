import pandas as pd
import numpy as np
import argparse
import os
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import linregress

# 获取文本嵌入
def get_embeddings(texts, tokenizer, model):
    batch_size = 128
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Processing embeddings"):
        batch = texts[i:i + batch_size]
        tokens = tokenizer.batch_encode_plus(batch, padding="max_length", max_length=35, truncation=True, return_tensors="pt")
        tokens_cuda = {k: v.cuda() for k, v in tokens.items()}
        output = model(**tokens_cuda)
        cls_rep = output[0][:, 0, :]
        embeddings.append(cls_rep.cpu().detach().numpy())

    return np.concatenate(embeddings, axis=0)

# 计算距离和相似度
def calculate_distances_and_similarities(source_texts, synonyms, tokenizer, model):
    source_embeddings = get_embeddings(source_texts, tokenizer, model)
    synonym_embeddings = get_embeddings(synonyms, tokenizer, model)

    dist = cdist(source_embeddings, synonym_embeddings, metric='euclidean')
    cos_sim = cosine_similarity(source_embeddings, synonym_embeddings)
    return dist, cos_sim

def plot_quadrant_chart(df, save_prefix):

    levels = df['Synonym_Level'].unique()
    plt.figure(figsize=(12, 12))

    for i, level in enumerate(levels):
        plt.subplot(2, 2, i + 1)
        subset = df[df['Synonym_Level'] == level]
        x = subset['distance']
        y = subset['cosine_similarity']

        plt.scatter(x, y, alpha=0.6)

        # 拟合回归线
        slope, intercept, r_value, _, _ = linregress(x, y)
        reg_line = slope * x + intercept
        plt.plot(x, reg_line, color='orange', label=f'Regression line (R²={r_value**2:.2f})')

        plt.axhline(y=0.5, color='r', linestyle='--')
        plt.axvline(x=1.0, color='r', linestyle='--')
        plt.xlabel('Distance')
        plt.ylabel('Cosine Similarity')
        plt.title(f'{level} Distance vs Cosine Similarity')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    save_path = f"{save_prefix}"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved combined figure: {save_path}")


def get_source_texts(mode, lbls, defs, comments):
    if mode == 'desc2id':
        return defs
    elif mode == 'id2id':
        return lbls
    elif mode == 'comment2id':
        return comments
    else:
        raise ValueError(f"Invalid mode: {mode}")
    

def main(args):
    df = pd.read_csv(args.input)

    # 判断字段是否存在
    required_columns = ['Synonym_Level', 'lbl', 'meta_definition_val', 'meta_comments', 'Synonym_Vocab']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # 根据模式选择要去除缺失值的列
    if args.mode == 'desc2id':
        drop_col = 'meta_definition_val'
    elif args.mode == 'id2id':
        drop_col = 'lbl'
    elif args.mode == 'comment2id':
        drop_col = 'meta_comments'
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    # 去除缺失值，目标列里的缺失值
    df = df.dropna(subset=[drop_col, 'Synonym_Vocab'])

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).cuda()

    grouped = df.groupby('Synonym_Level')
    all_results = []

    for level, group in grouped:
        meta_definition_vals = group['meta_definition_val'].tolist()
        meta_comments_vals = group['meta_comments'].tolist()
        lbl_vals = group['lbl'].tolist()
        synonyms = group['Synonym_Vocab'].tolist()

        # 安全获取 source_texts
        source_texts = get_source_texts(args.mode, lbl_vals, meta_definition_vals, meta_comments_vals)

        dist, cos_sim = calculate_distances_and_similarities(source_texts, synonyms, tokenizer, model)

        for i in range(len(source_texts)):
            all_results.append({
                'Synonym_Level': level,
                'lbl': lbl_vals[i],
                'meta_definition_val': meta_definition_vals[i],
                'meta_comments': meta_comments_vals[i],
                'Synonym_Vocab': synonyms[i],
                'distance': dist[i][i],
                'cosine_similarity': cos_sim[i][i]
            })

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(args.output, index=False)
    print(f"Results saved to '{args.output}'")

    # 保存图像
    plot_quadrant_chart(result_df, args.fig_path)



# 解析参数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate similarity between terms and synonyms")
    parser.add_argument('-i', '--input', required=True, help='Path to input CSV file')
    parser.add_argument('-o', '--output', required=True, help='Path to output CSV file')
    parser.add_argument('-m', '--mode', required=True, choices=['desc2id', 'id2id', 'comment2id'],
                        help='Comparison mode: desc2id / lbl2id / comment2id')
    parser.add_argument('--model', required=True, help='Path to pretrained HuggingFace model')
    parser.add_argument('--fig_path', required=True, help='Prefix path for saving plots, e.g., ./figs/plot')

    args = parser.parse_args()
    main(args)


# python script.py \
#   -i synonyms.csv \
#   -o results.csv \
#   -m desc2id \
#   --model /workspace/pretrain/SapBERT-from-PubMedBERT-fulltext \
#   --fig_path ./figures/desc2id_plot
