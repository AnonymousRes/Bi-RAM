#! /bin/bash


DATA=/workspace/lra/imdb-4000
SAVE_ROOT=/workspace/lra_saved_biram/imdb
exp_name=imdb_biram
SAVE=${SAVE_ROOT}/${exp_name}
SAVE_LOG=/workspace/Bi-RAM/out_log
rm -rf ${SAVE}
rm -f ${SAVE_LOG}/${exp_name}_log.txt
mkdir -p ${SAVE}

model=biram_lra_imdb
export CUDA_VISIBLE_DEVICES=1

python -u /workspace/Bi-RAM/train.py ${DATA} \
   --seed 2026 \
   --distributed-world-size 1 --ddp-backend c10d --find-unused-parameters \
   -a ${model} --task lra-text --input-type text \
   --k-times 1 \
   --encoder-layers 4 --encoder-attention-heads 8 --encoder-embed-dim 128 --encoder-ffn-embed-dim 128 \
   --activation-fn 'silu' --attention-activation-fn 'softmax' \
   --norm-type 'scalenorm' --sen-rep-type 'mp' --encoder-normalize-before \
   --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
   --optimizer adam --lr 0.0008 --adam-betas '(0.9, 0.99)' --adam-eps 1e-8 --clip-norm 1 --clip-mode 'total' \
   --dropout 0.2 --act-dropout 0.2 --weight-decay 0.06 \
   --batch-size 90 --sentence-avg --update-freq 1 --required-batch-size-multiple 1 \
   --lr-scheduler 'fixed' --max-epoch 200 \
   --keep-last-epochs 1 \
   --max-sentences-valid 25 \
   --save-dir ${SAVE} --log-format simple --log-interval 300 --num-workers 0 | tee -a ${SAVE_LOG}/${exp_name}_log.txt

python /workspace/Bi-RAM/fairseq_cli/validate.py ${DATA} --task lra-text --batch-size 64 --valid-subset test --path ${SAVE}/checkpoint_best.pt | tee -a ${SAVE_LOG}/${exp_name}_log.txt
