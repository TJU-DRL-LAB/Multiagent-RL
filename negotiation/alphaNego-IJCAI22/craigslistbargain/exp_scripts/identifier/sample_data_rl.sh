USE_GPU=$1
SEED="0"
if [ $# -ge 2 ]; then
  SEED=$2
fi
EXP_NAME="hard_pmask_7_rl_"${SEED}
# tom_itentity_test
mkdir checkpoint/${EXP_NAME}
PYTHONPATH=. python multi_rl.py --schema-path data/craigslist-schema.json \
--scenarios-path data/train-scenarios.json \
--valid-scenarios-path data/dev-scenarios.json \
--price-tracker data/price_tracker.pkl \
--agent-checkpoints checkpoint/a2c_0.0001_0/model_reward-0.0865_e1850.pt checkpoint/language/model_best.pt \
--model-path checkpoint/hard_pmask_rl --mappings mappings/language \
--optim adam --rnn-type RNN --rnn-size 300 --max-grad-norm -1 \
--agents pt-neural pt-neural-r \
--report-every 50 --max-turns 20 --num-dialogues 8960 \
--sample --temperature 0.5 --max-length 20 --reward margin \
--dia-num 20 --state-length 4 --epochs 2 --use-utterance \
--model lf2lf --model-type a2c --tom-test --seed ${SEED} \
--learning-rate 0.001 --name ${EXP_NAME} --hidden-depth 1 ${USE_GPU}
