#!/bin/bash

# USE THE FOLLOWING COMMAND TO RUN THIS SCRIPT IN A DOCKER CONTAINER WITH THE NECESSARY DEPENDENCIES INSTALLED
#
# docker run -it --rm \
#   -v ${PWD}:/workspace \
#   -w /workspace \
#   betty1202/pytorch:allennlp_202105 \
#   bash
#

PRE_MODEL_PATH=./Table2Charts/Table2Charts/Results/Models/best-excel.pt
DF_PATH=./Table2Charts/Table2Charts/Data/Example/data/0.t0.DF.json
EMB_PATH=./Table2Charts/Table2Charts/Data/Example/embeddings/fasttext/0.EMB.json

python ./Table2Charts/Table2Charts/single_inference.py \
  --df_path $DF_PATH \
  --emb_path $EMB_PATH \
  --model_path $PRE_MODEL_PATH