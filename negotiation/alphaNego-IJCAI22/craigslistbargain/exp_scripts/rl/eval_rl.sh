EXP_NAME="eval_rl"
USE_GPU=$1
SEED="0"
BETA="1"
if [ $# -ge 2 ]; then
  SEED=$2
fi
if [ $# -ge 3 ]; then
  BETA=$3
fi
CHECK_POINT="checkpoint/a2c/model_best.pt"
if [ $# -ge 4 ]; then
  CHECK_POINT=$4
fi

echo "beta:"${BETA}

mkdir checkpoint/${EXP_NAME}
PYTHONPATH=. python multi_rl.py --schema-path data/craigslist-schema.json \
--scenarios-path data/train-scenarios.json \
--valid-scenarios-path data/dev-scenarios.json \
--price-tracker data/price_tracker.pkl \
--agent-checkpoints ${CHECK_POINT} checkpoint/language/model_best.pt \
--model-path checkpoint/${EXP_NAME} --mappings mappings/language \
--optim adam --tom-beta ${BETA} \
--agents pt-neural pt-neural-r \
--report-every 50 --max-turns 20 --num-dialogues 10000 \
--sample --temperature 0.5 --max-length 20 --reward margin \
--dia-num 20 --state-length 4 \
--model lf2lf --model-type a2c --name ${EXP_NAME} --num-cpus 1 \
--epochs 10 --gpuid ${USE_GPU} --batch-size 128 --seed ${SEED} --get-dialogues

#PYTHONPATH=. python multi_rl.py --schema-path data/craigslist-schema.json \
#--scenarios-path data/train-scenarios.json \
#--valid-scenarios-path data/dev-scenarios.json \
#--price-tracker data/price_tracker.pkl \
#--agent-checkpoints checkpoint/a2c_0.001_0/model_reward-0.3950_e350.pt checkpoint/language/model_best.pt \
#--model-path checkpoint/tom_inf --mappings mappings/language \
#--optim adam --learning-rate 0.001 \
#--agents tom pt-neural-r --load-identity-from checkpoint/uttr_id_tom_history_7_4/model_best.pt \
#--report-every 50 --max-turns 20 --num-dialogues 10000 \
#--sample --temperature 0.5 --max-length 20 --reward margin \
#--dia-num 20 --state-length 4 \
#--model lf2lf --model-type a2c --name tom_inf --num-cpus 5 \
#--epochs 2000 --gpuid 0 --batch-size 128 --seed 0 \
#--tom-hidden-size 128 --tom-hidden-depth 2 --id-hidden-size 128 --id-hidden-depth 2 \
#--strategy-in-words --tom-model uttr_id_history_tom