#!/bin/bash
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
cd $SCRIPT_DIR

cache_dir=/tmp/DeBERTa/

function setup_glue_data(){
	task=$1
	mkdir -p $cache_dir
	if [[ ! -e $cache_dir/glue_tasks/${task}/train.tsv ]]; then
		curl -J -L https://raw.githubusercontent.com/nyu-mll/jiant/v1.3.2/scripts/download_glue_data.py | python3 - --data_dir $cache_dir/glue_tasks --tasks $task
	fi
}

#init=xlarge-v2 
init=base
tokenizer_type=spm
tokenizer_model=spm 

tag=XLarge
Task=MNLI
#setup_glue_data $Task
#91.45/91.49
#$cache_dir/glue_tasks/$Task 
data=/mount/biglm/bert/glue/data/MNLI 
../utils/train.sh -i $init --config config.json -t $Task --data $data --tag $tag -o /tmp/ttonly/$tag/$task -- --num_train_epochs 4 --accumulative_update 1 --warmup 500 --learning_rate 1.5e-5 --train_batch_size 64 --max_seq_length 256 --dump 1000 --cls_drop 0.15 --fp16 True --max_grad_norm 1 #  --tokenizer_type $tokenizer_type --tokenizer_model $tokenizer_model  --seed 1234
