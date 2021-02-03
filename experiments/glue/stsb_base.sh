#!/bin/bash
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
cd $SCRIPT_DIR

cache_dir=data/

function setup_glue_data(){
	task=$1
	mkdir -p $cache_dir
	if [[ ! -e $cache_dir/glue_tasks/${task}/train.tsv ]]; then
		python3 ../../utils/download_glue_data.py --data_dir $cache_dir/glue_tasks
	fi
}

init=base 

tag=Base
Task=STS-B

setup_glue_data $Task
../utils/train.sh --config config.json  -t $Task --data $cache_dir/glue_tasks/$Task --tag $tag -i $init -o $cache_dir/ttonly/$tag/$task -- --num_train_epochs 6 --accumulative_update 1 --warmup 100 --learning_rate 2e-5 --train_batch_size 32 --max_seq_len 128
