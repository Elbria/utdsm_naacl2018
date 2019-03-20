#!/usr/bin/env bash

PROJECT_DIR=$PWD
VIRT_ENV=${PROJECT_DIR}/utdsm_venv
DATA_DIR=${PROJECT_DIR}/data

echo "Activate virtual environment"
source ${VIRT_ENV}/bin/activate
$VIRT_ENV/bin/pip install -r requirements.txt

python ${PROJECT_DIR}/model/utdsm.py \
        --corpus_doc ${DATA_DIR}/toy_corpus.doc.cln \
        --corpus_sent ${DATA_DIR}/toy_corpus.sent.cln\
        --size 100 \
        --window 5 \
        --cbow 0 \
        --topics 5 \
        --anchors 500 \
        --anchors-selection='unsupervised' \
        --output ${PROJECT_DIR}/outputs \





