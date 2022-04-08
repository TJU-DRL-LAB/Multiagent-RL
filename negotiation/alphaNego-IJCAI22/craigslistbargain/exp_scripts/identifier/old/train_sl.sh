USE_GPU=$1

mkdir checkpoint/language; mkdir mappings/language;
PYTHONPATH=. python main.py --schema-path data/craigslist-schema.json --train-examples-paths data/train-luis-post-new.json --test-examples-paths data/dev-luis-post-new.json \
--price-tracker data/price_tracker.pkl \
--model lf2lf \
--model-path checkpoint/language --mappings mappings/language \
--word-vec-size 300 --pretrained-wordvec '' '' \
--rnn-size 300 --rnn-type RNN --global-attention multibank_general \
--num-context 2 --stateful \
--batch-size 128  --optim adagrad --learning-rate 0.01 \
--epochs 20 --report-every 500 \
--cache cache/language --ignore-cache \
--verbose  --state-length 4 --dia-num 20 --use-utterance --hidden-size 64 ${USE_GPU}