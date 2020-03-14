#!/usr/bin/env bash

# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

############################################################################################################

base_dir=/data/$USER/bidirectional_translation
install_libs=false
source_lang=en
target_lang=de

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
dataset_dir=$base_dir/data/iwslt17.tokenized.$source_lang-$target_lang
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
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=$subword_nmt_dir/subword_nmt
BPE_TOKENS=10000
URL="https://wit3.fbk.eu/archive/2017-01-trnted/texts/$source_lang/$target_lang/$source_lang-$target_lang.tgz"
GZ=$source_lang-$target_lang.tgz

############################################################################################################

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

############################################################################################################

echo "Downloading data from ${URL}..."
cd $orig_dir
wget "$URL"

if [ -f $GZ ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

############################################################################################################

tar zxvf $GZ
cd ..

echo "pre-processing train data..."
for l in $source_lang $target_lang; do
    f=train.tags.$source_lang-$target_lang.$l
    tok=train.tags.$source_lang-$target_lang.tok.$l

    cat $orig_dir/$source_lang-$target_lang/$f | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\/description>//g' | \
    perl $TOKENIZER -threads 8 -l $l > $tmp_dir/$tok
    echo ""
done

perl $CLEAN -ratio 1.5 $tmp_dir/train.tags.$source_lang-$target_lang.tok $source_lang $target_lang $tmp_dir/train.tags.$source_lang-$target_lang.clean 1 175

for l in $source_lang $target_lang; do
    perl $LC < $tmp_dir/train.tags.$source_lang-$target_lang.clean.$l > $tmp_dir/train.tags.$source_lang-$target_lang.$l
done

############################################################################################################

echo "pre-processing valid/test data..."
for l in $source_lang $target_lang; do
    for o in `ls $orig_dir/$source_lang-$target_lang/IWSLT17.TED*.$l.xml`; do
    fname=${o##*/}
    f=$tmp_dir/${fname%.*}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -l $l | \
    perl $LC > $f
    echo ""
    done
done

############################################################################################################

echo "creating train, valid, test..."
for l in $source_lang $target_lang; do
    awk '{if (NR%23 == 0)  print $0; }' $tmp_dir/train.tags.$source_lang-$target_lang.$l > $tmp_dir/valid.$l
    awk '{if (NR%23 != 0)  print $0; }' $tmp_dir/train.tags.$source_lang-$target_lang.$l > $tmp_dir/train.$l

    cat $tmp_dir/IWSLT17.TED.dev2010.$source_lang-$target_lang.$l \
        $tmp_dir/IWSLT17.TED.tst2010.$source_lang-$target_lang.$l \
        $tmp_dir/IWSLT17.TED.tst2011.$source_lang-$target_lang.$l \
        $tmp_dir/IWSLT17.TED.tst2012.$source_lang-$target_lang.$l \
        $tmp_dir/IWSLT17.TED.tst2013.$source_lang-$target_lang.$l \
        $tmp_dir/IWSLT17.TED.tst2014.$source_lang-$target_lang.$l \
        $tmp_dir/IWSLT17.TED.tst2015.$source_lang-$target_lang.$l \
        > $tmp_dir/test.$l

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
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp_dir/$f > $prep_dir/$f
    done
done

############################################################################################################

fairseq-preprocess --source-lang $source_lang --target-lang $target_lang \
    --trainpref $prep_dir/train --validpref $prep_dir/valid --testpref $prep_dir/test \
    --destdir $dataset_dir

############################################################################################################

echo "removing tmp dirs..."

rm -r $prep_dir
rm -r $tmp_dir
rm -r $orig_dir

############################################################################################################
