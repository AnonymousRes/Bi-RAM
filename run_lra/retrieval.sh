#! /bin/bash


DATA=/workspace/lra/aan
SAVE_ROOT=/workspace/lra_saved_biram/aan
exp_name=aan_biram
SAVE=${SAVE_ROOT}/${exp_name}
SAVE_LOG=/workspace/Bi-RAM/out_log
rm -rf ${SAVE}
rm -f ${SAVE_LOG}/${exp_name}_log.txt
mkdir -p ${SAVE}

model=biram_lra_aan
export CUDA_VISIBLE_DEVICES=1

python -u /workspace/Bi-RAM/train.py ${DATA} \
   --seed 2026 \
   --distributed-world-size 1 --ddp-backend c10d --find-unused-parameters \
   -a ${model} --task lra-text --input-type text \
   --encoder-layers 6 --encoder-attention-heads 8 --encoder-embed-dim 128 --encoder-ffn-embed-dim 128 \
   --activation-fn 'silu' --attention-activation-fn 'softmax' \
   --norm-type 'scalenorm' --sen-rep-type 'mp' --encoder-normalize-before \
   --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
   --optimizer adam --lr 0.0015 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
   --dropout 0.2 --act-dropout 0.2 --weight-decay 0.02 \
   --batch-size 35 --sentence-avg --update-freq 1 \
   --lr-scheduler linear_decay --total-num-update 300000 --max-update 300000 --end-learning-rate 0.0 --warmup-updates 30000 --warmup-init-lr '1e-07' \
   --keep-last-epochs 1 --max-sentences-valid 16 \
   --save-dir ${SAVE} --log-format simple --log-interval 10000 --num-workers 0 | tee -a ${SAVE_LOG}/${exp_name}_log.txt

python /workspace/Bi-RAM/fairseq_cli/validate.py ${DATA} --task lra-text --batch-size 32 --valid-subset test --path ${SAVE}/checkpoint_best.pt | tee -a ${SAVE_LOG}/${exp_name}_log.txt
