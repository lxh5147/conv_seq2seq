export PYTHONIOENCODING=UTF-8

export DATA_PATH=~/nmt_data/nist_tokenized_zh-en

export VOCAB_SOURCE=${DATA_PATH}/vocab.zh
export VOCAB_TARGET=${DATA_PATH}/vocab.en
export TRAIN_SOURCES=${DATA_PATH}/train_src.shuffle
export TRAIN_TARGETS=${DATA_PATH}/train_trg.shuffle
export DEV_SOURCES=${DATA_PATH}/valid_src
export DEV_TARGETS=${DATA_PATH}/valid_trg0
export TEST_SOURCES=${DATA_PATH}/test_src
export TEST_TARGETS="${DATA_PATH}/test_trg0 ${DATA_PATH}/test_trg1 ${DATA_PATH}/test_trg2 ${DATA_PATH}/test_trg3"

export TRAIN_STEPS=1000000


export MODEL_DIR=${TMPDIR:-/tmp}/nmt_conv_seq2seq_zh-en
mkdir -p $MODEL_DIR

# Create Vocabulary

BASE_DIR="."

${BASE_DIR}/bin/tools/generate_vocab.py \
  < ${TRAIN_SOURCES} \
  > ${VOCAB_SOURCE}
echo "Wrote ${VOCAB_SOURCE}"

${BASE_DIR}/bin/tools/generate_vocab.py \
  < ${TRAIN_TARGETS} \
  > ${VOCAB_TARGET}
echo "Wrote ${VOCAB_TARGET}"


'''
nohup python -m bin.train \
  --config_paths="
      ./example_configs/conv_seq2seq.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipelineFairseq
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 80 \
  --eval_every_n_steps 5000 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR &

'''


export PRED_DIR=${MODEL_DIR}/pred
mkdir -p ${PRED_DIR}
'''
###with greedy search
python -m bin.infer \
  --tasks "
    - class: DecodeText" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 1 
    decoder.class: seq2seq.decoders.ConvDecoderFairseq" \
  --input_pipeline "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TEST_SOURCES" \
  > ${PRED_DIR}/predictions.txt

'''
###with beam search
python -m bin.infer \
  --tasks "
    - class: DecodeText
    - class: DumpBeams
      params:
        file: ${PRED_DIR}/beams.npz" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 5 
    decoder.class: seq2seq.decoders.ConvDecoderFairseqBS" \
  --input_pipeline "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TEST_SOURCES" \
  > ${PRED_DIR}/predictions.txt


./bin/tools/multi-bleu.perl ${TEST_TARGETS} < ${PRED_DIR}/predictions.txt



