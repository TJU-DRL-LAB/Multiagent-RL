EXP_NAME="eval_tom_fid"$1
USE_GPU=$2
SEED="0"
BETA="1"
if [ $# -ge 3 ]; then
  SEED=$3
fi
if [ $# -ge 4 ]; then
  BETA=$4
fi
RL_CHECK_POINT="checkpoint/language/model_best.pt"
if [ $# -ge 5 ]; then
  RL_CHECK_POINT=$5
fi

TOM_CHECK_POINT="--load-identity-from checkpoint/train_uttr_fid_history_tom_7_4/model_best.pt"
if [ $# -ge 6 ]; then
  TOM_CHECK_POINT="--load-identity-from $6"
fi
MODEL_NAME="uttr_fid_history_tom"

echo "beta:"${BETA}

mkdir checkpoint/${EXP_NAME}
PYTHONPATH=. python multi_rl.py --schema-path data/craigslist-schema.json \
--scenarios-path data/train-scenarios.json \
--valid-scenarios-path data/dev-scenarios.json \
--price-tracker data/price_tracker.pkl \
--agent-checkpoints ${RL_CHECK_POINT} checkpoint/language/model_best.pt \
--model-path checkpoint/${EXP_NAME} --mappings mappings/language \
--optim adam --tom-beta ${BETA} \
--agents tom pt-neural-r ${TOM_CHECK_POINT} \
--report-every 50 --max-turns 20 --num-dialogues 10000 \
--sample --temperature 0.5 --max-length 20 --reward margin \
--dia-num 20 --state-length 4 \
--model lf2lf --model-type a2c --name ${EXP_NAME} --num-cpus 5 \
--epochs 10 ${USE_GPU} --batch-size 128 --seed ${SEED} --get-dialogues --tom-model ${MODEL_NAME}

#PYTHONPATH=. python multi_rl.py --schema-path data/craigslist-schema.json \
#--scenarios-path data/train-scenarios.json \
#--valid-scenarios-path data/dev-scenarios.json \
#--price-tracker data/price_tracker.pkl \
#--agent-checkpoints checkpoint/a2c_0.001_0/model_reward-0.3950_e350.pt checkpoint/language/model_best.pt \
#--model-path checkpoint/tom_inf --mappings mappings/language \
#--optim adam --learning-rate 0.001 \
#--agents tom pt-neural-r --load-identity-from checkpoint/uttr_fid_tom_history_7_4/model_best.pt \
#--report-every 50 --max-turns 20 --num-dialogues 10000 \
#--sample --temperature 0.5 --max-length 20 --reward margin \
#--dia-num 20 --state-length 4 \
#--model lf2lf --model-type a2c --name tom_inf --num-cpus 5 \
#--epochs 2000 --gpuid 0 --batch-size 128 --seed 0 \
#--tom-hidden-size 128 --tom-hidden-depth 2 --id-hidden-size 128 --id-hidden-depth 2 \
#--strategy-in-words --tom-model uttr_fid_history_tom