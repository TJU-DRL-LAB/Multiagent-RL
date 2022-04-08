EXP_NAME="sampleh_uttr_history_tom_7"$1
USE_GPU=$2
SEED="0"
LR="0.001"
if [ $# -ge 3 ]; then
  SEED=$3
fi
if [ $# -ge 4 ]; then
  LR=$4
fi
LOAD_SAMPLE="--load-sample cache/hard_pmask_7_${SEED}/data.pkl"
if [ $# -ge 5 ]; then
  LOAD_SAMPLE="--load-sample cache/$5/data.pkl"
fi

TOM_CHECK_POINT="--load-identity-from checkpoint/uttr_history_tom_7_0/model_best.pt"
if [ $# -ge 6 ]; then
  TOM_CHECK_POINT="--load-identity-from $6"
fi

echo "load sample from ${LOAD_SAMPLE}"
MODEL_NAME="history"

mkdir checkpoint/${EXP_NAME}
PYTHONPATH=. python multi_rl.py --schema-path data/craigslist-schema.json \
--scenarios-path data/train-scenarios.json \
--valid-scenarios-path data/dev-scenarios.json \
--price-tracker data/price_tracker.pkl \
--agent-checkpoints checkpoint/language/model_best.pt checkpoint/language/model_best.pt \
--model-path checkpoint/${EXP_NAME} --mappings mappings/language \
--optim adam --rnn-type RNN --rnn-size 300 --max-grad-norm -1 \
--agents pt-neural pt-neural-r ${TOM_CHECK_POINT} \\
--report-every 50 --max-turns 20 --num-dialogues 20 \
--sample --temperature 0.5 --max-length 20 --reward margin \
--dia-num 20 --state-length 4 --epochs 1 --use-utterance \
--model lf2lf --model-type a2c --tom-test ${LOAD_SAMPLE} \
--learning-rate ${LR} --name ${EXP_NAME} --seed ${SEED} \
--tom-hidden-size 128 --tom-hidden-depth 2 \
--strategy-in-words --tom-model ${MODEL_NAME} ${USE_GPU} --only-run