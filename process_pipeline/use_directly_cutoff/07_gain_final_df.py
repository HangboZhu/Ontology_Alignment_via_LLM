import pandas as pd
import networkx as nx
import argparse

def process_synonyms(input_file, output_file, remove_self_match):
    df = pd.read_csv(input_file)

    df_self = df[df['cluster'] == 'self'].copy()
    df_nonself = df[df['cluster'] != 'self'].copy()

    df_self['query_node'] = (
        df_self['query_id'].fillna('').astype(str) + '|' +
        df_self['query_lbl'].fillna('') + '|' +
        df_self['query_description'].fillna('') + '|' +
        df_self.index.astype(str)
    )
    df_self['pred_node'] = pd.NA

    df_nonself['query_node'] = (
        df_nonself['query_id'].fillna('').astype(str) + '|' +
        df_nonself['query_lbl'].fillna('') + '|' +
        df_nonself['query_description'].fillna('') + '|' +
        df_nonself.index.astype(str)
    )
    df_nonself['pred_node'] = (
        df_nonself['predicted_id'].fillna('').astype(str) + '|' +
        df_nonself['predicted_lbl'].fillna('') + '|' +
        df_nonself['predicted_description'].fillna('')
    )

    # 构建图
    G = nx.Graph()
    all_nodes = pd.concat([df_nonself['query_node'], df_nonself['pred_node']]).dropna().unique().tolist()
    G.add_nodes_from(all_nodes)

    edges = df_nonself[['query_node', 'pred_node']].dropna().drop_duplicates().values.tolist()
    sym_edges = [[b, a] for a, b in edges if [b, a] not in edges]
    G.add_edges_from(edges + sym_edges)

    df_self['query_lbl'] = df_self['query_lbl'].fillna('')
    df = pd.concat([df_nonself, df_self], ignore_index=True)

    if remove_self_match:
        df = df[df['query_id'] != df['predicted_id']].copy()

    # 构建节点到标签集合映射
    node_to_labels = {}

    # 非self样本：连通分量映射标签
    for comp in nx.connected_components(G):
        labels = set()
        for node in comp:
            parts = node.split('|')
            if len(parts) > 1:
                labels.add(parts[1])
        for node in comp:
            node_to_labels[node] = labels

    # self样本：每个自己独立标签组
    for _, row in df_self.iterrows():
        node = row['query_node']
        lbl = row['query_lbl']
        node_to_labels[node] = {lbl}

    synonym_vocab_list = []
    synonyms_group_list = []
    for _, row in df.iterrows():
        query_node = row['query_node']
        query_lbl = row['query_lbl']
        all_labels = node_to_labels.get(query_node, set())
        synonyms = sorted(str(lbl) for lbl in all_labels if lbl != query_lbl)
        synonyms_group = sorted(str(lbl) for lbl in all_labels)
        synonym_vocab_list.append('  ,'.join(synonyms))
        synonyms_group_list.append('  ,'.join(synonyms_group))

    df['Synonym_Vocab'] = synonym_vocab_list
    df['Synonym_Group'] = synonyms_group_list

    output_cols = [
        'query_id', 'query_lbl', 'query_description',
        'predicted_id', 'predicted_lbl', 'predicted_description',
        'distance', 'similarity', 'predicted_synonym_level',
        'synonym_similarity_mean', 'cluster',
        'Synonym_Vocab', 'Synonym_Group'
    ]

    df[output_cols].to_csv(output_file, index=False)
    print(f"✅ Done. Output saved to: {output_file}")

# 9. CLI 调用支持
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group synonyms and add Synonym_Vocab.")
    parser.add_argument("-i", "--input", required=True, help="Input CSV file path")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file path")
    parser.add_argument("-rm", "--remove-self-match", type=str, default="false", choices=["true", "false"], help="Remove rows where query_id == predicted_id (true/false)")

    args = parser.parse_args()
    remove_self_match = args.remove_self_match.lower() == "true"
    process_synonyms(args.input, args.output, remove_self_match)

    # input_file = "./data/out_data/cl/lbl2lbl_result.csv"  # Replace with your input file path
    # output_file = 'test.csv'  # Replace with your desired output file path
    # process_synonyms(input_file, output_file)