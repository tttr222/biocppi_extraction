PIPELINE=pipeline_test
INPUT=$PIPELINE/pipeline_feed.txt
PTOKEN=$PIPELINE/pipeline1.tokenized.txt
PNER=$PIPELINE/pipeline2.ner.txt
PNER2=$PIPELINE/pipeline2.5.ner.txt
PGNORM=$PIPELINE/pipeline3.gnorm.txt
OUTPUT=$PIPELINE/pipeline_output.txt
DATAPATH=corpus_train
GNCACHE=gene_normalization

CUDA_VISIBLE_DEVICES=""
python -u $PIPELINE/tokenize_input.py < $INPUT > $PTOKEN
python -u ner_model/annotate.py --datapath=$DATAPATH < $PTOKEN > $PNER
python -u ner_correction/annotate.py --datapath=$DATAPATH < $PNER > $PNER2
python -u gn_model/annotate.py --datapath=$DATAPATH --cachepath=$GNCACHE < $PNER2 > $PGNORM
python -u rc_model/extract.py --datapath=$DATAPATH < $PGNORM > $OUTPUT
echo "Output saved to $OUTPUT" 
