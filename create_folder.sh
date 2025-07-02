#!/bin/bash

# 定义五种实体类型
sources=("cl" "doid" "go" "hp" "so")

# 创建data根目录
mkdir -p data

# 创建 csv_file/raw 下子目录
for src in "${sources[@]}"; do
  mkdir -p "data/csv_file/raw/$src"
done

mkdir -p "data/csv_file/raw/fusion_all"

# 创建 json_structure 下子目录
for src in "${sources[@]}"; do
  mkdir -p "data/json_structure/$src"
done

# 创建 out_data 下子目录
for src in "${sources[@]}"; do
  mkdir -p "data/out_data/$src/use_directly_cutoff"
  mkdir -p "data/out_data/$src/use_synonym_info"
done

# 创建 raw_data 下子目录
for src in "${sources[@]}"; do
  mkdir -p "data/raw_data/$src"
done

mkdir -p "data/out_data/fusion_all"

# 创建 synonym_data 下子目录
for src in "${sources[@]}"; do
  mkdir -p "data/synonym_data/synonym_format_data/$src"
  mkdir -p "data/synonym_data/synonym_similarity_data/$src"
  mkdir -p "data/synonym_data/synonym_similarity_summary_data/$src"
done

# 创建 images 文件夹及子目录（与 data 同级）
mkdir -p images
for src in "${sources[@]}"; do
  mkdir -p "images/$src"
done

echo "✅ 所有文件夹创建完成！"

