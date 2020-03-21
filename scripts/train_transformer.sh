#!/usr/bin/env bash

##############################################################################################################

current_dir="$(cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"
base_dir=/data/$USER/bidirectional_translation
install_libs=false
prepare_dataset=false
run_tensorboard=false
source_lang=en
target_lang=fr
device_id=0
port=6006

##############################################################################################################

usage="Usage: $PROG [options]\n\n
Options:\n
  --help\t\t\t\tPrint this message and exit\n
  --base_dir\t\t\tBase directory (default=$base_dir).\n
  --install-libs\t\t\tInstall libs (default=$install_libs).\n
  --prepare-dataset\t\tPrepare dataset (default=$prepare_dataset).\n
  --run-tensorboard\t\tRun tensorboard (default=$run_tensorboard).\n
  --source-lang\t\t\tSource language (default=$source_lang).\n
  --target-lang\t\t\tTarget language (default=$target_lang).\n
  --device-id\t\t\tCuda device ID (default=$device_id).\n
  --port\t\t\t\tTensorboard port (default=$port).\n
";

##############################################################################################################

while [ $# -gt 0 ]; do
    case "${1# *}" in  # ${1# *} strips any leading spaces from the arguments
        --help) echo -e $usage; exit 0 ;;
        --base-dir)
            shift; base_dir="$1"; shift ;;
        --install-libs)
            shift; if [ "$1" == "true" ] || [ "$1" == "false" ]; then install_libs=$1; shift; else install_libs=true; fi ;;
        --prepare-dataset)
            shift; if [ "$1" == "true" ] || [ "$1" == "false" ]; then prepare_dataset=$1; shift; else prepare_dataset=true; fi ;;
        --run-tensorboard)
            shift; if [ "$1" == "true" ] || [ "$1" == "false" ]; then run_tensorboard=$1; shift; else run_tensorboard=true; fi ;;
        --source-lang)
            shift; source_lang="$1"; shift ;;
        --target-lang)
            shift; target_lang="$1"; shift ;;
        --device-id)
            shift; device_id="$1"; shift ;;
        --port)
            shift; port="$1"; shift ;;
        -*)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
        *)   break ;;   # end of options: interpreted as num-leaves
    esac
done

##############################################################################################################

data_dir=$base_dir/data
libs_dir=$base_dir/libs
fairseq_dir=$libs_dir/fairseq
dataset=wmt14_${source_lang}-${target_lang}
dataset_dir=$base_dir/data/$dataset
logs_dir=$base_dir/logs
checkpoints_dir=$base_dir/checkpoints

##############################################################################################################

if [ ! -d "$base_dir" ]; then
    mkdir $base_dir
fi

if [ ! -d "$libs_dir" ]; then
    mkdir $libs_dir
fi

if [ ! -d "$logs_dir"   ]; then
    mkdir $logs_dir
fi

if [ ! -d "$checkpoints_dir" ]; then
    mkdir $checkpoints_dir
fi

if [ ! -d "$logs_dir/$dataset"    ]; then
    mkdir $logs_dir/$dataset
fi

if [ ! -d "$checkpoints_dir/$dataset"  ]; then
    mkdir $checkpoints_dir/$dataset
fi

##############################################################################################################

if [ ! -d $fairseq_dir ]; then
    echo -e "\n----------------------------------------------------------------------------------------------"
    echo -e "$(date +"%D %T") Downloading fairseq\n"

    git clone https://github.com/pytorch/fairseq $fairseq_dir
fi

##############################################################################################################

if [ $install_libs == true ]; then
    echo -e "\n----------------------------------------------------------------------------------------------"
    echo -e "$(date +"%D %T") Installing libs\n"

    pip install --upgrade pip --user
    pip install torchvision --upgrade --user
    pip install torch --upgrade --user
    pip install $fairseq_dir --user
    pip install sacremoses --user
    pip install tensorboardX --user
    pip install tensorboard --user
fi

##############################################################################################################

if [ $prepare_dataset == true ]; then
    echo -e "\n----------------------------------------------------------------------------------------------"
    echo -e "$(date +"%D %T") Preparing dataset\n"

    bash $current_dir/prepare_dataset.sh \
        --base-dir $base_dir \
        --source-lang $source_lang \
        --target-lang $target_lang
fi

##############################################################################################################

if [ $run_tensorboard == true ]; then
    echo -e "\n----------------------------------------------------------------------------------------------"
    echo -e "$(date +"%D %T") Running tensorboard\n"

    tensorboard \
        --logdir $logs_dir/$dataset \
        --port $port \
        --bind_all \
        &

    sleep 10
fi

##############################################################################################################

echo -e "\n----------------------------------------------------------------------------------------------"
echo -e "$(date +"%D %T") Training transformer\n"

export CUDA_VISIBLE_DEVICES=$device_id

fairseq-train $dataset_dir \
    --arch transformer \
    --share-decoder-input-output-embed \
    --max-tokens 4096 \
    --attention-dropout 0.1 \
    --encoder-embed-dim  512 --decoder-embed-dim 512 \
    --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
    --label-smoothing 0.1 \
    --layernorm-embedding \
    --encoder-normalize-before --decoder-normalize-before \
    --encoder-attention-heads 8 --encoder-attention-heads 8 \
    --encoder-layers 6 --decoder-layers 6 \
    --dropout 0.1 \
    --warmup-updates 16000 \
    --optimizer adam --adam-betas '(0.9, 0.998)' --clip-norm 0 --adam-eps 1e-09 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --save-dir $checkpoints_dir/$dataset \
    --save-interval 1 \
    --tensorboard-logdir $logs_dir/$dataset \
    --lr 5e-4 \
    --lr-scheduler inverse_sqrt \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --valid-subset valid,test

##############################################################################################################
