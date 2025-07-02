#!/bin/bash

# 设置输入输出路径和模型路径
INPUT_CSV="../../data/csv_file/raw/fusion_all/combined_nodes_flatten.csv"
OUTPUT_DIR="../../data/out_data/fusion_all/"
MODEL_PATH="/workspace/pretrain/SapBERT-from-PubMedBERT-fulltext/"
BATCH_SIZE=1024
METHOD="all"
SIM_THRESHOLD=0.85

# Step 1: 运行 embedding + 匹配脚本
python 06_oncology_fusion.py \
  -i "$INPUT_CSV" \
  -o "$OUTPUT_DIR" \
  -c $SIM_THRESHOLD \
  --model "$MODEL_PATH" \
  -b $BATCH_SIZE \
  -m $METHOD

# Step 2: 生成最终结果，去除 self-match
python 07_gain_final_df.py \
  -i "${OUTPUT_DIR}/lbl2lbl_result.csv" \
  -o "${OUTPUT_DIR}/fusion_final_remove_self_match.csv" \
  -rm true

# Step 3: 生成包含 self-match 的结果
python 07_gain_final_df.py \
  -i "${OUTPUT_DIR}/lbl2lbl_result.csv" \
  -o "${OUTPUT_DIR}/fusion_final_contain_self_match.csv" \
  -rm false
