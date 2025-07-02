#!/bin/bash

list=(cl doid go hp so)

for index in "${!list[@]}"; do
  cl="${list[$index]}"
  echo "Processing $cl (index $index)..."

  python 00_konw_json_structure.py \
    -i ../../data/raw_data/${cl}/${cl}.json \
    -o ../../data/json_structure/${cl}/${cl}_json_structure.txt

  python 01_json_2_csv_4_nodes.py \
    -i ../../data/raw_data/${cl}/${cl}.json \
    -o ../../data/csv_file/raw/${cl}/${cl}_nodes_flatten.csv

  python 06_oncology_fusion.py \
    -i ../../data/csv_file/raw/${cl}/${cl}_nodes_flatten.csv \
    -o ../../data/out_data/${cl}/use_directly_cutoff \
    -c 0.85 \
    --model /workspace/pretrain/SapBERT-from-PubMedBERT-fulltext/ \
    -b 1024 \
    -m all

  python 07_gain_final_df.py \
    -i ../../data/out_data/${cl}/use_directly_cutoff/lbl2lbl_result.csv\
    -o ../../data/out_data/${cl}/use_directly_cutoff/${cl}_final.csv \

done

