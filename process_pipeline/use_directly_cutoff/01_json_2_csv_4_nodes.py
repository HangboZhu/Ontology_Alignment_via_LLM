# 提取json文件中的nodes信息到一个CSV文件中
import json
from collections import defaultdict
import pandas as pd
import argparse


def merge_keys_tree(obj, tree):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key not in tree:
                tree[key] = {}
            merge_keys_tree(value, tree[key])
    elif isinstance(obj, list):
        for item in obj:
            merge_keys_tree(item, tree)  # 合并所有元素结构

def print_tree(tree, indent=0):
    for key, subtree in tree.items():
        print('  ' * indent + f"- {key}")
        print_tree(subtree, indent + 1)

def flatten(d, parent_key='', sep='_'):
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten(v, new_key, sep=sep).items())
    elif isinstance(d, list):
        if all(isinstance(i, dict) for i in d):
            combined = {}
            for i in d:
                flat = flatten(i, '', sep)
                for fk, fv in flat.items():
                    combined.setdefault(fk, []).append(str(fv))
            for ck, cv in combined.items():
                items.append((f"{parent_key}{sep}{ck}", '%AND%'.join(cv)))
        else:
            items.append((parent_key, '%AND%'.join(map(str, d))))
    else:
        items.append((parent_key, d))
    return dict(items)
def main(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    data = data.get("graphs", [])
    
    # 可选结构分析输出
    structure_tree = {}
    for entry in data:
        merge_keys_tree(entry, structure_tree)
    
    print("The structure of the JSON data is:")
    print_tree(structure_tree)
    
    all_flattened_nodes = []
    for entry in data:
        for node in entry.get("nodes", []):
            flat_node = {}
            for key, value in node.items():
                # 只提取了·meta· 这个分支下边的节点信息，因为只用于nodes的主体融合
                if key != "meta":
                    flat_node[key] = value
                else:
                    flat_node.update(flatten(value, parent_key="meta")) 
            all_flattened_nodes.append(flat_node)

    df = pd.DataFrame(all_flattened_nodes)
    df["id"] = df["id"].apply(lambda x: x.split("/")[-1] if isinstance(x, str) else x)
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"🤓CSV file '{output_file}' has been created successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flatten nested 'nodes' from JSON and save to CSV.")
    parser.add_argument("-i", "--input", required=True, help="Path to input JSON file")
    parser.add_argument("-o", "--output", required=True, help="Path to output CSV file")
    args = parser.parse_args()

    main(args.input, args.output)
    
    
