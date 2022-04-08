EXP_NAME="eval_agent_human"$1
USE_GPU=$2
SEED="0"
BETA="0.05"
if [ $# -ge 3 ]; then
  SEED=$3
fi
if [ $# -ge 4 ]; then
  BETA=$4
fi
RL_CHECK_POINT="checkpoint/uttr_id_tom_history/model_best.pt"
if [ $# -ge 5 ]; then
  RL_CHECK_POINT=$5
fi

TOM_CHECK_POINT="--load-identity-from checkpoint/uttr_id_tom_history/model_best.pt"
if [ $# -ge 6 ]; then
  TOM_CHECK_POINT="--load-identity-from $6"
fi
MODEL_NAME="uttr_id_history_tom"

echo "beta:"${BETA}

mkdir checkpoint/${EXP_NAME}
PYTHONPATH=. python multi_rl.py --schema-path data/craigslist-schema.json \
--scenarios-path data/train-scenarios.json \
--valid-scenarios-path data/dev-scenarios.json \
--price-tracker data/price_tracker.pkl \
--agent-checkpoints ${RL_CHECK_POINT} checkpoint/language/model_best.pt \
--model-path checkpoint/${EXP_NAME} --mappings mappings/language \
--optim adam --tom-beta ${BETA} \
--agents pt-neural-dsac pt-neural-s ${TOM_CHECK_POINT} \
--report-every 50 --max-turns 20 --num-dialogues 10000 \
--sample --temperature 0.5 --max-length 20 --reward margin \
--dia-num 20 --state-length 4 \
--model lf2lf --model-type a2c --name ${EXP_NAME} --num-cpus 5 \
--epochs 10 --gpuid ${USE_GPU} --batch-size 128 --seed ${SEED} --get-dialogues --tom-model ${MODEL_NAME} \
--load-type from_rl \
--actor-path checkpoint/dsac_learnsp_seed2_alpha0.1_policylr0_kl0.1_margin_0.8sl_sp/actor_seed2_200.pth \
--zf1-path checkpoint/dsac_learnsp_seed0_alpha0.1_policylr0.0000001_kl0_margin_0.8sl_fsp/zf1_seed0_750.pth \
--zf2-path checkpoint/dsac_learnsp_seed0_alpha0.1_policylr0.0000001_kl0_margin_0.8sl_fsp/zf2_seed0_750.pth \
