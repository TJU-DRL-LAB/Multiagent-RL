EXP_NAME="a2c"
USE_GPU=$1
SEED="0"
LR="0.0001"
if [ $# -ge 2 ]; then
  SEED=$2
fi
if [ $# -ge 3 ]; then
  LR=$3
fi

echo "LR:"${LR}
#DEBUG="--debug"

mkdir checkpoint/${EXP_NAME}
PYTHONPATH=. python multi_rl.py --schema-path data/craigslist-schema.json \
--scenarios-path data/train-scenarios.json \
--valid-scenarios-path data/dev-scenarios.json \
--price-tracker data/price_tracker.pkl \
--agent-checkpoints checkpoint/language/model_best.pt checkpoint/language/model_best.pt \
--model-path checkpoint/${EXP_NAME} --mappings mappings/language \
--optim adam --learning-rate ${LR} \
--agents pt-neural pt-neural-r \
--report-every 50 --max-turns 20 --num-dialogues 10000 \
--sample --temperature 0.5 --max-length 20 --reward base_utility \
--dia-num 20 --state-length 4 \
--model lf2lf --model-type a2c --name ${EXP_NAME} --num-cpus 1 \
--epochs 2000 --gpuid ${USE_GPU} --batch-size 128 --seed ${SEED} ${DEBUG}