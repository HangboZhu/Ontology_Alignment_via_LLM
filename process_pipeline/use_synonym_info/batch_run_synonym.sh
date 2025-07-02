#!/bin/bash

declare -A name_map=(
  [cl]="Cell Ontology"
  [doid]="Disease Ontology"
  [go]="Gene Ontology"
  [hp]="Human Phenotype"
  [so]="Sequence Ontology"
)

list=(cl doid go hp so)

for cl in "${list[@]}"; do
  echo "Processing $cl..."

  # Step 1
  python 00_konw_json_structure.py \
    -i ../../data/raw_data/${cl}/${cl}.json \
    -o ../../data/json_structure/${cl}/${cl}_json_structure.txt

  # Step 2
  python 01_json_2_csv_4_nodes.py \
    -i ../../data/raw_data/${cl}/${cl}.json \
    -o ../../data/csv_file/raw/${cl}/${cl}_nodes_flatten.csv

  # Step 3
  python 03_process_synonym.py \
    -i ../../data/csv_file/raw/${cl}/${cl}_nodes_flatten.csv \
    -o ../../data/synonym_data/synonym_format_data/${cl}/ \
    -m all

  # Step 4
  for mode in id2id desc2id comment2id; do
    python 04_calculate_synonym.py \
      -i ../../data/synonym_data/synonym_format_data/${cl}/processed_lbl_mode.csv \
      -o ../../data/synonym_data/synonym_similarity_data/${cl}/${mode}_synonym_calcaulate.csv \
      -m ${mode} \
      --model /workspace/pretrain/SapBERT-from-PubMedBERT-fulltext \
      --fig_path ../../images/${cl}/${mode}_distance_similarity_regression.png
  done

  # Step 5
  python 05_synonym_summary.py \
    -i1 ../../data/synonym_data/synonym_similarity_data/${cl}/desc2id_synonym_calcaulate.csv \
    -i2 ../../data/synonym_data/synonym_similarity_data/${cl}/id2id_synonym_calcaulate.csv \
    -i3 ../../data/synonym_data/synonym_similarity_data/${cl}/comment2id_synonym_calcaulate.csv \
    -s ../../data/synonym_data/synonym_similarity_summary_data/${cl}/${cl}_synonym_calculate_summary.csv \
    -op ../../images/${cl}/ \
    -n "${name_map[$cl]}"

  # Step 6
  python 07_oncology_fusion_use_syno.py \
    -i ../../data/csv_file/raw/${cl}/${cl}_nodes_flatten.csv \
    -s ../../data/synonym_data/synonym_similarity_data/${cl}/id2id_synonym_calcaulate.csv \
    --model /workspace/pretrain/SapBERT-from-PubMedBERT-fulltext/ \
    -o ../../data/out_data/${cl}/use_synonym_info/ \
    -b 1024 \
    -m all
done

