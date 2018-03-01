PIPELINE=pipeline_test
INPUT=$PIPELINE/pipeline_feed.txt
PTOKEN=$PIPELINE/pipeline1.tokenized.txt
PNER=$PIPELINE/pipeline2.ner.txt
PNER2=$PIPELINE/pipeline2.5.ner.txt
PGNORM=$PIPELINE/pipeline3.gnorm.txt
OUTPUT=$PIPELINE/pipeline_output.txt
DATAPATH=corpus_train
GNCACHE=gn_model

CUDA_VISIBLE_DEVICES=""
echo "Step 1 / Tokenizing input feed / $PTOKEN"
python -u $PIPELINE/tokenize_input.py < $INPUT > $PTOKEN
echo "Step 2 / NER Annotations / $PNER"
python -u ner_model/annotate.py --datapath=$DATAPATH < $PTOKEN > $PNER
echo "Step 3 / NER Corrections / $PNER2"
python -u ner_correction/annotate.py --datapath=$DATAPATH < $PNER > $PNER2
echo "Step 4 / Gene Normalization / $PGNORM"
python -u gn_model/annotate.py --datapath=$DATAPATH --cachepath=$GNCACHE < $PNER2 > $PGNORM
echo "Step 5 / PPIm Extraction / $OUTPUT"
python -u rc_model/extract.py --datapath=$DATAPATH < $PGNORM > $OUTPUT
echo "Done"