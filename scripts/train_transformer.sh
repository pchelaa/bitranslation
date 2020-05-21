#!/usr/bin/env bash

##############################################################################################################

current_dir="$(cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"
base_dir=/data/$USER/courses/bt
install_libs=false
prepare_dataset=false
merge_dataset=false
clean_checkpoints=false
run_tensorboard=false
tensorboard_port=6006
max_tokens=2048
kl_init_steps=15000
kl_warmup_steps=5000
save_interval=5
source_lang=en
target_lang=fr
valid_subset=test
device_id=0
criterion=label_smoothed_cross_entropy
arch=transformer
embed_dim=512
ffn_embed_dim=2048
even_source=0
even_target=0
group_by_first_token=0
max_source_positions=1024
max_target_positions=1024
lang_tokens="@@@@,^^^^"

##############################################################################################################

usage="Usage: $PROG [options]\n\n
Options:\n
  --help\t\t\t\tPrint this message and exit\n
  --base_dir\t\t\tBase directory (default=$base_dir).\n
  --install-libs\t\t\tInstall libs (default=$install_libs).\n
  --prepare-dataset\t\tPrepare dataset (default=$prepare_dataset).\n
  --merge-dataset\t\tCreate merged dataset (default=$merge_dataset).\n
  --clean-checkpoints\t\tClean checkpoints (default=$clean_checkpoints).\n
  --run-tensorboard\t\tRun tensorboard (default=$run_tensorboard).\n
  --tensorboard-port\t\tTensorboard port (default=$tensorboard_port).\n
  --max-tokens\t\t\tMax tokens (default=$max_tokens).\n
  --kl-init-steps\t\tKL init steps (default=$kl_init_steps).\n
  --kl-warmup-steps\t\tKL warmup steps (default=$kl_warmup_steps).\n
  --save-interval\t\tSave interval (default=$save_interval).\n
  --source-lang\t\t\tSource language (default=$source_lang).\n
  --target-lang\t\t\tTarget language (default=$target_lang).\n
  --valid-subset\t\t\tValid subset (default=$valid_subset).\n
  --device-id\t\t\tCuda device ID (default=$device_id).\n
  --criterion\t\t\tCriterion (default=$criterion).\n
  --arch\t\t\t\tAcrchitecture (default=$arch).\n
  --model\t\t\tModel name.\n
  --embed-dim\t\t\tEmbedding dimension (default=$embed_dim).\n
  --ffn-embed-dim\t\tFfn embedding dimension (default=$ffn_embed_dim).\n
  --even-source\t\t\tMake number of tokens even for source dataset (default=$even_source).\n
  --even-target\t\t\tMake number of tokens even for target dataset (default=$even_target).\n
  --group-by-first-token\t\tGroup batch by first token (default=$group_by_first_token).\n
  --max-source-positions\t\tMax source positions (default=$max_source_positions).\n
  --max-target-positions\t\tMax target positions (default=$max_target_positions).\n
  --lang-tokens\t\t\tComma separated special language tokens (default=$lang_tokens).\n
  --preset\t\t\tSettings preset (optionally).\n
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
        --merge-dataset)
            shift; if [ "${1}" == "true" ] || [ "${1}" == "false" ]; then merge_dataset=${1}; shift; else merge_dataset=true; fi ;;
        --clean-checkpoints)
            shift; if [ "${1}" == "true" ] || [ "${1}" == "false"  ]; then clean_checkpoints=${1}; shift; else clean_checkpoints=true; fi ;;
        --run-tensorboard)
            shift; if [ "$1" == "true" ] || [ "$1" == "false" ]; then run_tensorboard=$1; shift; else run_tensorboard=true; fi ;;
        --tensorboard-port)
            shift; tensorboard_port="$1"; shift ;;
        --max-tokens)
            shift; max_tokens="$1"; shift ;;
        --kl-init-steps)
            shift; kl_init_steps="$1"; shift ;;
        --kl-warmup-steps)
            shift; kl_warmup_steps="$1"; shift ;;
        --save-interval)
            shift; save_interval="$1"; shift ;;
        --source-lang)
            shift; source_lang="$1"; shift ;;
        --target-lang)
            shift; target_lang="$1"; shift ;;
        --valid-subset)
            shift; valid_subset="$1"; shift ;;
        --device-id)
            shift; device_id="$1"; shift ;;
        --criterion)
            shift; criterion="$1"; shift ;;
        --arch)
            shift; arch="$1"; shift ;;
        --model)
            shift; model="$1"; shift ;;
        --embed-dim)
            shift; embed_dim="$1"; shift ;;
        --ffn-embed-dim)
            shift; ffn_embed_dim="$1"; shift ;;
        --even-source)
            shift; even_source="$1"; shift ;;
        --even-target)
            shift; even_target="$1"; shift ;;
        --group-by-first-token)
            shift; group_by_first_token="$1"; shift ;;
        --max-source-positions)
            shift; max_source_positions="$1"; shift ;;
        --max-target-positions)
            shift; max_target_positions="$1"; shift ;;
        --lang-tokens)
            shift; lang_tokens="$1"; shift ;;
        --preset)
            shift; preset="$1"; shift ;;
        -*)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
        *)   break ;;   # end of options: interpreted as num-leaves
    esac
done

##############################################################################################################

if [ $preset == "base" ]; then
    arch=transformer
    criterion=label_smoothed_cross_entropy
    valid_subset=test
    embed_dim=512
    ffn_embed_dim=2048
    max_tokens=3072
    kl_init_steps=12000
    kl_warmup_steps=4000
    even_source=0
    even_target=0
    group_by_first_token=0
    max_source_poisitions=1024
    max_target_positions=1024
fi

if [ $preset == "flow" ]; then
    arch=transformer_with_flow
    criterion=label_smoothed_cross_entropy_with_kl
    valid_subset=test
    embed_dim=512
    ffn_embed_dim=2048
    max_tokens=3072
    kl_init_steps=12000
    kl_warmup_steps=4000
    even_source=1
    even_target=1
    group_by_first_token=0
    max_source_positions=1024
    max_target_positions=1024
fi

if [ $preset == "multi-decoder" ]; then
    arch=transformer_multi_decoder
    criterion=label_smoothed_cross_entropy
    valid_subset=en-fr,fr-en
    embed_dim=512
    ffn_embed_dim=2048
    max_tokens=4096
    kl_init_steps=0
    kl_warmup_steps=0
    even_source=0
    even_target=0
    group_by_first_token=1
    max_source_poisitions=1024
    max_target_positions=1024
fi

if [ $preset == "multi-flow" ]; then
    arch=transformer_multi_flow
    criterion=label_smoothed_cross_entropy_with_kl
    valid_subset=en-fr,fr-en
    embed_dim=512
    ffn_embed_dim=2048
    max_tokens=3072
    kl_init_steps=12000
    kl_warmup_steps=4000
    even_source=1
    even_target=1
    group_by_first_token=1
    max_source_positions=1024
    max_target_positions=1024
fi


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

if [ ! -d "$logs_dir/$dataset"   ]; then
    mkdir $logs_dir/$dataset
fi

if [ ! -d "$checkpoints_dir" ]; then
    mkdir $checkpoints_dir
fi

if [ ! -d "$checkpoints_dir/$dataset"  ]; then
    mkdir $checkpoints_dir/$dataset
fi

if [ -z "$model" ]; then
    model=$dataset
fi

##############################################################################################################

if [ $clean_checkpoints == true ]; then
    echo -e "\n----------------------------------------------------------------------------------------------"
    echo -e "$(date +"%D %T") Downloading fairseq\n"

    rm -r $checkpoints_dir/$dataset/*
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

if [ $merge_dataset == true ]; then
    echo -e "\n----------------------------------------------------------------------------------------------"
    echo -e "$(date +"%D %T") Creating merged dataset\n"

    bash $current_dir/merge_dataset.sh \
        --base-dir $base_dir \
        --lang en --lang fr \
        --token @@@@ \
        --source-lang $source_lang \
        --target-lang $target_lang
fi

##############################################################################################################

if [ $run_tensorboard == true ]; then
    echo -e "\n----------------------------------------------------------------------------------------------"
    echo -e "$(date +"%D %T") Running tensorboard\n"

    tensorboard \
        --logdir $logs_dir \
        --port $tensorboard_port \
        --bind_all \
        &

    sleep 10
fi

##############################################################################################################

echo -e "\n----------------------------------------------------------------------------------------------"
echo -e "$(date +"%D %T") Training transformer\n"

export CUDA_VISIBLE_DEVICES=$device_id

/data/anjukirkov/courses/bt/fairseq/fairseq_cli/train.py $dataset_dir \
    --arch $arch \
    --share-decoder-input-output-embed \
    --max-tokens $max_tokens \
    --attention-dropout 0.1 \
    --encoder-embed-dim  $embed_dim --decoder-embed-dim $embed_dim \
    --encoder-ffn-embed-dim $ffn_embed_dim --decoder-ffn-embed-dim $ffn_embed_dim \
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
    --save-dir $checkpoints_dir/$model \
    --save-interval $save_interval \
    --tensorboard-logdir $logs_dir/$model \
    --lr 5e-4 \
    --lr-scheduler inverse_sqrt \
    --weight-decay 0.0001 \
    --criterion $criterion \
    --valid-subset $valid_subset \
    --kl-init-steps $kl_init_steps \
    --kl-warmup-steps $kl_warmup_steps \
    --even-source $even_source \
    --even-target $even_target \
    --group-by-first-token $group_by_first_token \
    --max-source-positions $max_source_positions \
    --max-target-positions $max_target_positions \
    --lang-tokens $lang_tokens \
    --ddp-backend=no_c10d \
    --skip-invalid-size-inputs-valid-test \
    --reset-optimizer

##############################################################################################################
