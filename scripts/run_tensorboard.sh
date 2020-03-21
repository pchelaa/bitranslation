#!/usr/bin/env bash

##############################################################################################################

current_dir="$(cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"
base_dir=/data/$USER/courses/bt
install_libs=false
source_lang=enfr
target_lang=fren
port=6016

##############################################################################################################

usage="Usage: $PROG [options]\n\n
Options:\n
  --help\t\t\t\tPrint this message and exit\n
  --base_dir\t\t\tBase directory (default=$base_dir).\n
  --install-libs\t\t\tInstall libs (default=$install_libs).\n
  --source-lang\t\t\tSource language (default=$source_lang).\n
  --target-lang\t\t\tTarget language (default=$target_lang).\n
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
        --source-lang)
            shift; source_lang="$1"; shift ;;
        --target-lang)
            shift; target_lang="$1"; shift ;;
        --port)
            shift; port="$1"; shift ;;
        -*)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
        *)   break ;;   # end of options: interpreted as num-leaves
    esac
done

##############################################################################################################

data_dir=$base_dir/data
libs_dir=$base_dir/libs
dataset=wmt14_${source_lang}-${target_lang}
dataset_dir=$base_dir/data/$dataset
logs_dir=$base_dir/logs

##############################################################################################################

if [ $install_libs == true ]; then
    echo -e "\n----------------------------------------------------------------------------------------------"
    echo -e "$(date +"%D %T") Installing libs\n"

    pip install tensorboardX --user
    pip install tensorboard --user
fi

##############################################################################################################

echo -e "\n----------------------------------------------------------------------------------------------"
echo -e "$(date +"%D %T") Running tensorboard\n"

tensorboard \
    --logdir $logs_dir/$dataset \
    --port $port \
    --bind_all \
    &

##############################################################################################################

