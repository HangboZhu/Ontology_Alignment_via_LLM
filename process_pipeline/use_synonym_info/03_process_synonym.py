# 这个脚本是根据处理好的csv文件，以其她提供的一个csv文件中的同义词
# 处理成一个lbl对应一个同义词的样式

import pandas as pd
import argparse
import os

def process_synonyms(input_path, output_path, mode):
    df = pd.read_csv(input_path)
    if mode == "def_mode":
        df = df[df["meta_definition_val"].notna()]
    elif mode == "lbl_mode":
        df = df[df["lbl"].notna()]

    base_columns = ['id', 'lbl', 'type', 'meta_definition_val', 'meta_comments',
                    'meta_subsets', 'meta_deprecated']

    processed_rows = []

    for _, row in df.iterrows():
        preds = str(row['meta_synonyms_pred']) if pd.notna(row['meta_synonyms_pred']) else ''
        vals = str(row['meta_synonyms_val']) if pd.notna(row['meta_synonyms_val']) else ''

        pred_list = preds.split('%AND%') if preds else []
        val_list = vals.split('%AND%') if vals else []

        if pred_list and val_list and len(pred_list) == len(val_list):
            for pred, val in zip(pred_list, val_list):
                new_row = row[base_columns].to_dict()
                new_row['Synonym_Level'] = pred
                new_row['Synonym_Vocab'] = val
                processed_rows.append(new_row)
        elif preds and vals:
            new_row = row[base_columns].to_dict()
            new_row['Synonym_Level'] = preds
            new_row['Synonym_Vocab'] = vals
            processed_rows.append(new_row)
        else:
            new_row = row[base_columns].to_dict()
            new_row['Synonym_Level'] = None
            new_row['Synonym_Vocab'] = None
            processed_rows.append(new_row)

    final_df = pd.DataFrame(processed_rows)
    # final_df["meta_definition_val"]去除是na的行。也就是没有同义词的行
    final_df = final_df[final_df["meta_definition_val"].notna()]
    final_df = final_df[final_df["meta_deprecated"] != True]
    final_df.to_csv(output_path, index=False)
    print(f"✅ Processed data saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process meta_synonyms from CSV.")
    parser.add_argument('-i', '--input', required=True, help="Path to input CSV file")
    parser.add_argument('-o', '--output', required=True, help="Path to output CSV Folder")
    parser.add_argument("-m", "--mode", choices=["def_mode", "lbl_mode", "all"], default="all",
                        help="Mode to filter data: def_mode (drop empty definitions), "
                             "lbl_mode (drop empty labels), all (keep all)")

    args = parser.parse_args()
    
    if args.mode == "def_mode":
        output_file = os.path.join(args.output, "processed_def_mode.csv")
        process_synonyms(args.input, output_file, args.mode)
    elif args.mode == "lbl_mode":
        output_file = os.path.join(args.output, "processed_lbl_mode.csv")
        process_synonyms(args.input, output_file, args.mode)
    elif args.mode == "all":
        for i in ["def_mode", "lbl_mode"]:
            output_file = os.path.join(args.output, f"processed_{i}.csv")
            process_synonyms(args.input, output_file, i)
        
    
