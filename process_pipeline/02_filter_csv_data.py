# 这个脚本的作用是从所有的csv文件中挑选出有用的字段，并进行一些处理。
# 还有deprecated字段的处理, 把已经废弃的字段从description中删除。
# 提供俩中模式继续筛选，一种是筛选description有定义的节点，另一种是筛选lal有值的节点。
# 筛选后续用得到的字段，meta_subsets 处理成列表形式。
import pandas as pd
import argparse

def process_meta_subsets(value):
    if pd.isna(value) or not isinstance(value, str):
        return []
    return [item.split("#")[-1] for item in value.split("%AND%") if item]

# Filter data based on mode
def filter_data(data, mode):
    # Remove deprecated entries
    data = data[data["meta_deprecated"] != True]
    mode = str(mode)

    if mode == "def_mode":
        data = data[~data["meta_definition_val"].isna()]
    elif mode == "lbl_mode":
        data = data[~data["lbl"].isna()]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Filter and process cl_nodes CSV data")
    parser.add_argument('-i', '--input', required=True, help='Path to input CSV file')
    parser.add_argument('-o', '--output', required=True, help='Path to output CSV file')
    parser.add_argument('-m', '--mode', choices=['def_mode', 'lbl_mode'], required=True,
                        help='Filtering mode: def_mode (drop empty definitions), lbl_mode (drop empty labels)')
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)

    # Select required columns
    selected_columns = ['id', 'lbl', 'type', 'meta_definition_val',
                        'meta_comments', 'meta_subsets',
                        'meta_xrefs_val', 'meta_synonyms_pred', 'meta_synonyms_val', 'meta_deprecated']
    df = df[selected_columns]

    df = filter_data(df, args.mode)

    # Process 'meta_subsets' column
    df["meta_subsets"] = df["meta_subsets"].apply(process_meta_subsets)

    # Save
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Filtered and saved to: {args.output} (mode: {args.mode})")

if __name__ == "__main__":
    main()
