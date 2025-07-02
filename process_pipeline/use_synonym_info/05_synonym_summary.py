import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
from PIL import Image

def run_similarity_analysis(input1, input2, input3, summary_csv, output_dir, ontology_name):

    os.makedirs(output_dir, exist_ok=True)

    # 读取并标注方法
    df1 = pd.read_csv(input1)
    df2 = pd.read_csv(input2)
    df3 = pd.read_csv(input3)
    df1['Method'] = 'desc_2_id'
    df2['Method'] = 'id_2_id'
    df3['Method'] = 'comment_2_id'

    df = pd.concat([df1, df2, df3], ignore_index=True)

    # 统计 summary
    summary = df.groupby(['Synonym_Level', 'Method']).agg(
        distance_mean=('distance', 'mean'),
        distance_std=('distance', 'std'),
        similarity_mean=('cosine_similarity', 'mean'),
        similarity_std=('cosine_similarity', 'std')
    ).reset_index()

    summary['distance_out'] = summary['distance_mean'].round(4).astype(str) + '±' + summary['distance_std'].round(4).astype(str)
    summary['similarity_out'] = summary['similarity_mean'].round(4).astype(str) + '±' + summary['similarity_std'].round(4).astype(str)
    summary.to_csv(summary_csv, index=False)
    print(f"Saved summary to: {summary_csv}")

    # 转为long格式
    df_long = pd.melt(df, id_vars=['Synonym_Level', 'lbl', 'meta_definition_val', 'meta_comments', 'Synonym_Vocab', 'Method'],
                      value_vars=['distance', 'cosine_similarity'], var_name='Metric', value_name='Value')

    palette = ["#015493", "#F4A99B", "#999999"]
    box_pairs = [
        (("hasExactSynonym", "desc_2_id"), ("hasExactSynonym", "id_2_id")),
        (("hasExactSynonym", "id_2_id"), ("hasExactSynonym", "comment_2_id")),
        (("hasNarrowSynonym", "desc_2_id"), ("hasNarrowSynonym", "id_2_id")),
        (("hasNarrowSynonym", "id_2_id"), ("hasNarrowSynonym", "comment_2_id")),
        (("hasBroadSynonym", "desc_2_id"), ("hasBroadSynonym", "id_2_id")),
        (("hasBroadSynonym", "id_2_id"), ("hasBroadSynonym", "comment_2_id")),
        (("hasRelatedSynonym", "desc_2_id"), ("hasRelatedSynonym", "id_2_id")),
        (("hasRelatedSynonym", "id_2_id"), ("hasRelatedSynonym", "comment_2_id")),
    ]

    # 分别绘图
    for metric in ['distance', 'cosine_similarity']:
        # 设置字体和标签大小
        plt.rcParams['font.family'] = ['Times New Roman']
        plt.rcParams["axes.labelsize"] = 18
        
        # 创建图形并设置1行2列布局
        plt.figure(figsize=(12, 6))
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        subset = df_long[df_long['Metric'] == metric]
        sns.barplot(x='Synonym_Level', y='Value', hue='Method', data=subset, ci="sd", capsize=.1,
                    errwidth=1, errcolor="k", palette=palette, edgecolor="k", linewidth=1, ax=ax)

        annotator = Annotator(ax, data=subset, x='Synonym_Level', y='Value', hue='Method', pairs=box_pairs)
        annotator.configure(test='t-test_ind', text_format='star', line_height=0.03, line_width=1)
        annotator.apply_and_annotate()

        ax.set_title(f"Synonymous Ontology {metric.capitalize()} of {str(ontology_name)}", fontsize=13)
        ax.set_xlabel("Synonym Level", fontsize=11)
        ax.set_ylabel(metric.capitalize(), fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.legend(loc='lower right', fontsize=8)

        out_path = os.path.join(output_dir, f"synonym_{metric}_compare.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=600)
        plt.close()
        print(f"Saved plot: {out_path}")

    # 合并两个图横排拼接（可选）
    img1 = Image.open(os.path.join(output_dir, "synonym_distance_compare.png"))
    img2 = Image.open(os.path.join(output_dir, "synonym_cosine_similarity_compare.png"))

    total_width = img1.width + img2.width
    max_height = max(img1.height, img2.height)
    combined = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width, 0))
    combined_path = os.path.join(output_dir, "synonym_combined.png")
    combined.save(combined_path)
    print(f"Saved combined plot: {combined_path}")




def main():
    parser = argparse.ArgumentParser(description="Plot and summarize similarity comparison.")
    parser.add_argument('-i1', '--input1', required=True, help='Path to desc2id CSV file')
    parser.add_argument('-i2', '--input2', required=True, help='Path to id2id CSV file')
    parser.add_argument('-i3', '--input3', required=True, help='Path to comment2id CSV file')
    parser.add_argument('-s', '--summary_csv', required=True, help='Output path for summary CSV')
    parser.add_argument('-op', '--output_plot_dir', required=True, help='Output directory for plots')
    parser.add_argument("-n", "--ontology_name", required=True, help="Name of the ontology")

    args = parser.parse_args()

    run_similarity_analysis(args.input1, args.input2, args.input3, args.summary_csv, args.output_plot_dir, args.ontology_name)

if __name__ == "__main__":
    main()
