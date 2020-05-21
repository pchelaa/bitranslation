#!/usr/bin/env bash

# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

############################################################################################################

base_dir=/data/$USER/courses/bt
install_libs=false
source_lang=enfr
target_lang=fren
skip_download_files=false
skip_unpack_files=false
skip_preprocess_train_data=false
skip_preprocess_test_data=false
skip_merge_data=false
skip_shuffle_data=false
skip_learn_bpe=false
skip_fairseq_preprocess=false
skip_remove_tmp_dirs=false


############################################################################################################

usage="Usage: $PROG [options]\n\n
Options:\n
  --help\t\t\t\tPrint this message and exit\n
  --base_dir\t\t\tBase directory (default=$base_dir).\n
  --install-libs\t\t\tInstall libs (default=$install_libs).\n
  --lang\t\t\t\tSource languages and the corresponding tokens (required).\n
  --source-lang\t\t\tSource language (default=$source_lang).\n
  --target-lang\t\t\tTarget language(default=$target_lang).\n
  --skip-download-files\t\tSkip download files (default=$skip_download_files).\n
  --skip-unpack-files\t\tSkip unpack files (default=$skip_unpack_files).\n
  --skip-preprocess-train-data\tSkip preprocess train data (default=$skip_preprocess_train_data).\n
  --skip-preprocess-test-data\tSkip preprocess test data (default=$skip_preprocess_test_data).\n
  --skip-merge-data\t\tSkip merge data (default=$skip_merge_data).\n
  --skip-shuffle-data\t\tSkip shuffle data (default=$skip_shuffle_data).\n
  --skip-learn-bpe\t\tSkip learn bpe (default=$skip_learn_bpe).\n
  --skip-fairseq-preprocess\tSkip fairseq preprocess (default=$skip_fairseq_preprocess).\n
  --skip-remove-tmp-dirs\t\tSkip remove temporary directories (default=$skip_remove_tmp_dirs).\n
";

############################################################################################################

while [ $# -gt 0 ]; do
    case "${1# *}" in  # ${1# *} strips any leading spaces from the arguments
        --help) echo -e $usage; exit 0 ;;
        --base-dir)
            shift; base_dir="$1"; shift ;;
        --install-libs)
            shift; if [ "$1" == "true" ] || [ "$1" == "false" ]; then install_libs=$1; shift; else install_libs=true; fi ;;
        --lang)
            shift; langs+=("$1"); shift; if [ "${1:0:2}" != "--"  ]; then tokens+=("$1"); shift; else tokens+=(""); fi ;;
        --source-lang)
            shift; source_lang="$1"; shift ;;
        --target-lang)
            shift; target_lang="$1"; shift ;;
        --skip-download-files)
            shift; if [ "${1}" == "true" ] || [ "${1}" == "false" ]; then skip_download_files=${1}; shift; else skip_download_files=true; fi ;;
        --skip-unpack-files)
            shift; if [ "${1}" == "true" ] || [ "${1}" == "false" ]; then skip_unpack_files=${1}; shift; else  skip_unpack_files=true; fi ;;
        --skip-preprocess-train-data)
            shift; if [ "${1}" == "true"  ] || [ "${1}" == "false"  ]; then skip_preprocess_train_data=${1}; shift; else skip_preprocess_train_data=true; fi ;;
        --skip-preprocess-test-data)
            shift; if [ "${1}" == "true"  ] || [ "${1}" == "false"  ]; then skip_preprocess_test_data=${1}; shift; else skip_preprocess_test_data=true; fi ;;
        --skip-merge-data)
             shift; if [ "${1}" == "true"  ] || [ "${1}" == "false"  ]; then skip_merge__data=${1};         shift; else skip_merge_data=true; fi ;;
        --skip-shuffle-data)
            shift; if [ "${1}" == "true"  ] || [ "${1}" == "false"  ]; then skip_shuffle_data=${1}; shift; else skip_shuffle_data=true; fi ;;
        --skip-learn-bpe)
            shift; if [ "${1}" == "true"  ] || [ "${1}" == "false"  ]; then skip_learn_bpe=${1}; shift; else skip_learn_bpe=true; fi ;;
        --skip-fairseq-preprocess)
            shift; if [ "${1}" == "true"  ] || [ "${1}" == "false"  ]; then skip_fairseq_preprocess=${1}; shift; else skip_fairseq_preprocess=true; fi ;;
        --skip-remove-tmp-dirs)
            shift; if [ "${1}" == "true"  ] || [ "${1}" == "false"  ]; then skip_remove_tmp_dirs=${1}; shift; else skip_remove_tmp_dirs=true; fi ;;
        -*)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
        *)   break ;;   # end of options: interpreted as num-leaves
    esac
done

############################################################################################################

data_dir=$base_dir/data
libs_dir=$base_dir/libs
orig_dir=$data_dir/orig
dataset_dir=$data_dir/wmt14_$source_lang-$target_lang
prep_dir=$dataset_dir/prep
src_dir=$dataset_dir/src
merged_dir=$dataset_dir/merged
tmp_dir=$dataset_dir/tmp
fairseq_dir=$libs_dir/fairseq
mosesdecoder_dir=$libs_dir/mosesdecoder
subword_nmt_dir=$libs_dir/subword-nmt

############################################################################################################

if [ ! -d "$base_dir" ]; then
    mkdir $base_dir
fi

if [ ! -d "$libs_dir" ]; then
    mkdir $libs_dir
fi

if [ ! -d "$data_dir" ]; then
    mkdir $data_dir
fi

if [ -d "$dataset_dir" ]; then
    rm -r $dataset_dir
fi

mkdir -p $orig_dir $src_dir $tmp_dir $prep_dir $merged_dir $dataset_dir

############################################################################################################

if [ ! -d $fairseq_dir ]; then
    echo -e "\n--------------------------------------------------------------------------------------------"
    echo -e "$(date +"%D %T") Cloning Fireseq github repository...\n"

    git clone https://github.com/pytorch/fairseq $fairseq_dir
fi

if [ ! -d $mosesdecoder_dir ]; then
    echo -e "\n--------------------------------------------------------------------------------------------"
    echo -e "$(date +"%D %T") Cloning Moses github repository (for tokenization scripts)...\n"

    git clone https://github.com/moses-smt/mosesdecoder.git $mosesdecoder_dir
fi

if [ ! -d $subword_nmt_dir ]; then
    echo -e "\n--------------------------------------------------------------------------------------------"
    echo -e "$(date +"%D %T") Cloning Subword NMT repository (for BPE pre-processing)...\n"

    git clone https://github.com/rsennrich/subword-nmt.git $subword_nmt_dir
fi

############################################################################################################

if [ $install_libs == true ]; then
    echo -e "\n--------------------------------------------------------------------------------------------"
    echo -e "$(date +"%D %T") Installing libs\n"

    pip install --upgrade pip --user
    pip install torchvision --upgrade --user
    pip install torch --upgrade --user
    pip install $fairseq_dir --user
    pip install sacremoses --user
fi

############################################################################################################

SCRIPTS=$mosesdecoder_dir/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=$subword_nmt_dir/subword_nmt
BPE_TOKENS=6000

URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://statmt.org/wmt13/training-parallel-un.tgz"
    "http://statmt.org/wmt14/training-parallel-nc-v9.tgz"
    "http://statmt.org/wmt10/training-giga-fren.tar"
    "http://statmt.org/wmt14/test-full.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-un.tgz"
    "training-parallel-nc-v9.tgz"
    "training-giga-fren.tar"
    "test-full.tgz"
)
CORPORA=(
    "training/europarl-v7.fr-en"
    "commoncrawl.fr-en"
    "un/undoc.2000.fr-en"
    "training/news-commentary-v9.fr-en"
    "giga-fren.release2.fixed"
)
TESTS=(
    "test-full/newstest2014-fren"
)

############################################################################################################

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

############################################################################################################

if [ $skip_download_files == false ]; then
    echo -e "\n--------------------------------------------------------------------------------------------"
    echo -e "$(date +"%D %T") Downloading files\n"

    cd $orig_dir

    for ((i=0;i<${#URLS[@]};++i)); do
        file=${FILES[i]}
        if [ -f $file ]; then
            echo "$file already exists, skipping download"
        else
            url=${URLS[i]}
            wget "$url"
            if [ -f $file ]; then
                echo "$url successfully downloaded."
            else
                echo "$url not successfully downloaded."
                exit -1
            fi
        fi
    done

    cd ..
fi

############################################################################################################

if [ $skip_unpack_files == false ]; then
    echo -e "\n--------------------------------------------------------------------------------------------"
    echo -e "$(date +"%D %T") Unpacking files\n"

    for ((i=0;i<${#URLS[@]};++i)); do
        file=$orig_dir/${FILES[i]}

        if [ ${file: -4} == ".tgz"  ]; then
            tar zxvf $file --directory $src_dir
        elif [ ${file: -4} == ".tar"  ]; then
            tar xvf $file --directory $src_dir
        fi
    done

    gunzip $src_dir/giga-fren.release2.fixed.*.gz

    mv $src_dir/test-full/newstest2014-fren-src.en.sgm $src_dir/test-full/newstest2014-fren.en.sgm
    mv $src_dir/test-full/newstest2014-fren-ref.fr.sgm $src_dir/test-full/newstest2014-fren.fr.sgm

    for l in ${langs[@]}; do
        for f in "${CORPORA[@]}"; do
            awk '{if (NR%62 == 0)  print $0; }' $src_dir/$f.$l > $src_dir/$f.$l.filtered
            rm -f $src_dir/$f.$l
            mv $src_dir/$f.$l.filtered $src_dir/$f.$l
        done
    done
fi


############################################################################################################

if [ $skip_preprocess_train_data == false ]; then
    echo -e "\n--------------------------------------------------------------------------------------------"
    echo -e "$(date +"%D %T") Preprocessing train data\n"

    for l in ${langs[@]}; do
        rm -f  $tmp_dir/train.tags.en-fr.tok.$l

        for f in "${CORPORA[@]}"; do
            cat $src_dir/$f.$l | \
                perl $NORM_PUNC $l | \
                perl $REM_NON_PRINT_CHAR | \
                perl $TOKENIZER -threads 16 -a -l $l >> $tmp_dir/train.tags.en-fr.tok.$l
        done
    done

    for l in ${langs[@]}; do
        awk '{if (NR%23 == 0)  print $0; }' $tmp_dir/train.tags.en-fr.tok.$l > $tmp_dir/valid.$l
        awk '{if (NR%23 != 0)  print $0; }' $tmp_dir/train.tags.en-fr.tok.$l > $tmp_dir/train.$l
    done
fi

############################################################################################################

if [ $skip_preprocess_test_data == false ]; then
    echo -e "\n--------------------------------------------------------------------------------------------"
    echo -e "$(date +"%D %T") Preprocessing test data\n"

    for l in ${langs[@]}; do
        rm -f $tmp_dir/test.$l

        for f in "${TESTS[@]}"; do
            grep '<seg id' $src_dir/$f.$l.sgm | \
                sed -e 's/<seg id="[0-9]*">\s*//g' | \
                sed -e 's/\s*<\/seg>\s*//g' | \
                sed -e "s/\’/\'/g" | \
            perl $TOKENIZER -threads 16 -a -l $l >> $tmp_dir/test.$l
            echo ""
        done
    done
fi

############################################################################################################

if [ $skip_merge_data == false ]; then
    echo -e "\n--------------------------------------------------------------------------------------------"
    echo -e "$(date +"%D %T") Merging data\n"

    for f in train valid; do
        rm -f $merged_dir/$f.$source_lang

        for ((i=0;i<${#langs[@]};++i)); do
            L=${langs[i]}
            token=${tokens[i]}

            if [ ! -z "$token" ]; then
                awk -v token=$token '{ print token" "$0; }' $tmp_dir/$f.$L >> $merged_dir/$f.$source_lang
            else
                awk '{ print $0; }' $tmp_dir/$f.$L >> $merged_dir/$f.$source_lang
            fi
        done
    done

    for f in train valid; do
        rm -f $merged_dir/$f.$target_lang
        for ((i=${#langs[@]};i>0;--i)); do
             L=${langs[i-1]}
             awk '{print $0; }' $tmp_dir/$f.$L >> $merged_dir/$f.$target_lang
        done
    done

    for ((i=0;i<${#langs[@]};++i)); do
        L=${langs[i]}
        token=${tokens[i]}

        L1=$L
        if [ $L == fr ]; then
            L2=en
        else
            L2=fr
        fi

        if  [ ! -z "$token" ]; then
            awk -v token=$token '{ print token" "$0; }' $tmp_dir/test.$L > $merged_dir/$L1-$L2.$source_lang
        else
            awk '{ print $0; }' $tmp_dir/test.$L > $merged_dir/$L1-$L2.$source_lang
        fi
        awk '{ print $0; }' $tmp_dir/test.$L > $merged_dir/$L2-$L1.$target_lang
    done
fi

############################################################################################################

if [ $skip_shuffle_data == false ]; then
    for f in train valid en-fr fr-en; do
        for l in $source_lang $target_lang; do
            shuf $merged_dir/$f.$l --random-source=<(openssl enc -aes-256-ctr -pass pass:"42" --nosalt </dev/zero 2>/dev/null) >$merged_dir/$f.$l.shuffled
            rm -f $merged_dir/$f.$l
            mv $merged_dir/$f.$l.shuffled $merged_dir/$f.$l
        done
    done
fi

############################################################################################################

if [ $skip_learn_bpe == false ]; then
    echo -e "\n--------------------------------------------------------------------------------------------"
    echo -e "$(date +"%D %T") Learning bpe\n"

    TRAIN=$merged_dir/train.$source_lang-$target_lang
    BPE_CODE=$prep_dir/code
    rm -f $TRAIN
    for l in $source_lang $target_lang; do
       cat $merged_dir/train.$l >> $TRAIN
    done

    python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

    for L in $source_lang $target_lang; do
        for f in train.$L valid.$L en-fr.$L fr-en.$L; do
            echo "apply_bpe.py to ${f}..."
            python $BPEROOT/apply_bpe.py -c $BPE_CODE < $merged_dir/$f > $merged_dir/bpe.$f
        done
    done

    perl $CLEAN -ratio 1.5 $merged_dir/bpe.train $source_lang $target_lang $prep_dir/train 1 250
    perl $CLEAN -ratio 1.5 $merged_dir/bpe.valid $source_lang $target_lang $prep_dir/valid 1 250

    for L in $source_lang $target_lang; do
        cp $merged_dir/bpe.en-fr.$L $prep_dir/en-fr.$L
        cp $merged_dir/bpe.fr-en.$L $prep_dir/fr-en.$L
    done
fi

############################################################################################################

if [ $skip_fairseq_preprocess == false ]; then
    echo -e "\n--------------------------------------------------------------------------------------------"
    echo -e "$(date +"%D %T") Fairseq preprocessing\n"

    fairseq-preprocess --source-lang $source_lang --target-lang $target_lang \
        --trainpref $prep_dir/train --validpref $prep_dir/valid --testpref $prep_dir/en-fr,$prep_dir/fr-en \
        --destdir $dataset_dir

    mv $dataset_dir/test.$source_lang-$target_lang.$target_lang.idx $dataset_dir/en-fr.$source_lang-$target_lang.$target_lang.idx
    mv $dataset_dir/test.$source_lang-$target_lang.$source_lang.idx $dataset_dir/en-fr.$source_lang-$target_lang.$source_lang.idx
    mv $dataset_dir/test.$source_lang-$target_lang.$target_lang.bin $dataset_dir/en-fr.$source_lang-$target_lang.$target_lang.bin
    mv $dataset_dir/test.$source_lang-$target_lang.$source_lang.bin $dataset_dir/en-fr.$source_lang-$target_lang.$source_lang.bin

    mv $dataset_dir/test1.$source_lang-$target_lang.$target_lang.idx $dataset_dir/fr-en.$source_lang-$target_lang.$target_lang.idx
    mv $dataset_dir/test1.$source_lang-$target_lang.$source_lang.idx $dataset_dir/fr-en.$source_lang-$target_lang.$source_lang.idx
    mv $dataset_dir/test1.$source_lang-$target_lang.$target_lang.bin $dataset_dir/fr-en.$source_lang-$target_lang.$target_lang.bin
    mv $dataset_dir/test1.$source_lang-$target_lang.$source_lang.bin $dataset_dir/fr-en.$source_lang-$target_lang.$source_lang.bin
fi

############################################################################################################

if [ $skip_remove_tmp_dirs == false ]; then
    echo -e "\n--------------------------------------------------------------------------------------------"
    echo -e "$(date +"%D %T") Removing temporary directories\n"

    rm -r $prep_dir $tmp_dir $merged_dir $src_dir
fi

############################################################################################################
