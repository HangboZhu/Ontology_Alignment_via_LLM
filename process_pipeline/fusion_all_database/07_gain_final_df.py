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
    df_self['pred_node'] = pd.NA  # self 无连接

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

    # 4. 构建图
    G = nx.Graph()
    all_nodes = pd.concat([df_nonself['query_node'], df_nonself['pred_node']]).dropna().unique().tolist()
    G.add_nodes_from(all_nodes)
    edges = df_nonself[['query_node', 'pred_node']].dropna().drop_duplicates().values.tolist()
    G.add_edges_from(edges)

    # 5. 生成 group_id（图连通分量）
    node_to_group = {}
    group_to_nodes = {}
    group_counter = 0

    for comp in nx.connected_components(G):
        for node in comp:
            node_to_group[node] = group_counter
        group_to_nodes[group_counter] = comp
        group_counter += 1

    df_nonself['group'] = df_nonself['query_node'].map(node_to_group)

    df_self['group'] = range(group_counter, group_counter + len(df_self))
    df_self['query_lbl'] = df_self['query_lbl'].fillna('')
    # df_self.to_csv('self_grouped.csv', index=False)
    # df_nonself.to_csv('nonself_grouped.csv', index=False)

    df = pd.concat([df_nonself, df_self], ignore_index=True)
    
    
    # 根据参数决定是否移除 query_id == predicted_id 的行
    if remove_self_match:
        df = df[df['query_id'] != df['predicted_id']].copy()

    group_label_map = {}
    grouped = df.groupby('group')

    for group_id, group_df in grouped:
        labels = set(group_df['query_lbl'].dropna()) | set(group_df['predicted_lbl'].dropna())
        group_label_map[group_id] = labels

    synonym_vocab_list = []
    for i, row in df.iterrows():
        group_id = row['group']
        query_lbl = row['query_lbl']
        all_labels = group_label_map.get(group_id, set())
        synonyms = sorted(str(lbl) for lbl in all_labels if lbl != query_lbl)
        synonym_vocab = '  ,'.join(synonyms)
        synonym_vocab_list.append(synonym_vocab)

    df['Synonym_Vocab'] = synonym_vocab_list

    output_cols = [
        'query_id', 'query_lbl', 'query_description', "query_source",
        'predicted_id', 'predicted_lbl', 'predicted_description', "predicted_source",
        'distance', 'similarity', 'predicted_synonym_level',
        'synonym_similarity_mean', 'cluster',
        'group', 'Synonym_Vocab'
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

    # input_file = "/workspace/code/data/out_data/fusion_all/lbl2lbl_result.csv"  # Replace with your input file path
    # output_file = 'test.csv'  # Replace with your desired output file path
    # remove_self_match = True  # Set to True if you want to remove self-matching rows
    # process_synonyms(input_file, output_file, remove_self_match)