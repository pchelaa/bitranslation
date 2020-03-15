#!/usr/bin/env bash

# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

############################################################################################################

base_dir=/data/$USER/bidirectional_translation
install_libs=false
source_lang=en
target_lang=fr

############################################################################################################

usage="Usage: $PROG [options]\n\n
Options:\n
  --help\t\t\t\tPrint this message and exit\n
  --base_dir\t\t\tBase directory (default=$base_dir).\n
  --install-libs\t\t\tInstall libs (default=$install_libs).\n
  --source-lang\t\t\tSource language (default=$source_lang).\n
  --target-lang\t\t\tTarget language(default=$target_lang).\n
";

############################################################################################################

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
        -*)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
        *)   break ;;   # end of options: interpreted as num-leaves
    esac
done

############################################################################################################

data_dir=$base_dir/data
libs_dir=$base_dir/libs
dataset_dir=$base_dir/data/wmt14_$source_lang-$target_lang
prep_dir=$dataset_dir/prep
tmp_dir=$dataset_dir/tmp
orig_dir=$dataset_dir/orig
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

mkdir -p $orig_dir $tmp_dir $prep_dir $dataset_dir

############################################################################################################

if [ ! -d $fairseq_dir ]; then
    echo 'Cloning Fireseq github repository...'
    git clone https://github.com/pytorch/fairseq $fairseq_dir
fi

if [ ! -d $mosesdecoder_dir ]; then
    echo 'Cloning Moses github repository (for tokenization scripts)...'
    git clone https://github.com/moses-smt/mosesdecoder.git $mosesdecoder_dir
fi

if [ ! -d $subword_nmt_dir ]; then
    echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
    git clone https://github.com/rsennrich/subword-nmt.git $subword_nmt_dir
fi

############################################################################################################

if [ $install_libs == true ]; then
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
BPE_TOKENS=4000

URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://statmt.org/wmt13/training-parallel-un.tgz"
    "http://statmt.org/wmt14/training-parallel-nc-v9.tgz"
    "http://statmt.org/wmt10/training-giga-${target_lang}${source_lang}.tar"
    "http://statmt.org/wmt14/test-full.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-un.tgz"
    "training-parallel-nc-v9.tgz"
    "training-giga-${target_lang}${source_lang}.tar"
    "test-full.tgz"
)
CORPORA=(
    "training/europarl-v7.${target_lang}-${source_lang}"
    "commoncrawl.${target_lang}-${source_lang}"
    "un/undoc.2000.${target_lang}-${source_lang}"
    "training/news-commentary-v9.${target_lang}-${source_lang}"
    "giga-${target_lang}${source_lang}.release2.fixed"
)

############################################################################################################

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

############################################################################################################

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
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        fi
    fi
done

gunzip giga-${target_lang}${source_lang}.release2.fixed.*.gz
cd ..

############################################################################################################

echo "pre-processing train data..."
for l in $source_lang $target_lang; do
    rm $tmp_dir/train.tags.$source_lang-$target_lang.tok.$l
    for f in "${CORPORA[@]}"; do
        cat $orig_dir/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $l >> $tmp_dir/train.tags.$source_lang-$target_lang.tok.$l
    done
done

############################################################################################################

echo "pre-processing test data..."
for l in $source_lang $target_lang; do
    if [ "$l" == "$source_lang" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig_dir/test-full/newstest2014-${target_lang}${source_lang}-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -a -l $l > $tmp_dir/test.$l
    echo ""
done

############################################################################################################

echo "splitting train and valid..."
for l in $source_lang $target_lang; do
    awk '{if (NR%1333 == 0)  print $0; }' $tmp_dir/train.tags.$source_lang-$target_lang.tok.$l > $tmp_dir/valid.$l
    awk '{if (NR%1333 != 0)  print $0; }' $tmp_dir/train.tags.$source_lang-$target_lang.tok.$l > $tmp_dir/train.$l
done

TRAIN=$tmp_dir/train.$source_lang-$target_lang
BPE_CODE=$prep_dir/code
rm -f $TRAIN
for l in $source_lang $target_lang; do
    cat $tmp_dir/train.$l >> $TRAIN
done

############################################################################################################

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $source_lang $target_lang; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done

perl $CLEAN -ratio 1.5 $tmp_dir/bpe.train $source_lang $target_lang $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp_dir/bpe.valid $source_lang $target_lang $prep/valid 1 250

for L in $source_lang $target_lang; do
    cp $tmp_dir/bpe.test.$L $prep_dir/test.$L
done

############################################################################################################

fairseq-preprocess --source-lang $source_lang --target-lang $target_lang \
    --trainpref $prep_dir/train --validpref $prep_dir/valid --testpref $prep_dir/test \
    --destdir $dataset_dir

############################################################################################################

echo "removing tmp dirs..."

#rm -r $prep_dir
#rm -r $tmp_dir
#rm -r $orig_dir

############################################################################################################
