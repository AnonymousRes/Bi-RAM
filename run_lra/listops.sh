#! /bin/bash


DATA=/workspace/lra/listops
SAVE_ROOT=/workspace/lra_saved_biram/listops
exp_name=listops_biram
SAVE=${SAVE_ROOT}/${exp_name}
SAVE_LOG=/workspace/Bi-RAM/out_log
rm -rf ${SAVE}
rm -f ${SAVE_LOG}/${exp_name}_log.txt
mkdir -p ${SAVE}

model=biram_lra_listop
export CUDA_VISIBLE_DEVICES=1

python -u /workspace/Bi-RAM/train.py ${DATA} \
   --seed 2026 \
   --distributed-world-size 1 --ddp-backend c10d --find-unused-parameters \
   -a ${model} --task lra-text --input-type text \
   --encoder-layers 6 --encoder-attention-heads 8 --encoder-embed-dim 160 --encoder-ffn-embed-dim 160 \
   --activation-fn 'silu' --attention-activation-fn 'softmax' \
   --norm-type 'layernorm' --sen-rep-type 'mp' --encoder-normalize-before \
   --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
   --optimizer adam --lr 0.0006 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
   --dropout 0.2 --act-dropout 0.2 --weight-decay 0.05\
   --batch-size 64 --sentence-avg --update-freq 1 --max-sentences-valid 64 \
   --lr-scheduler fixed --max-epoch 200 \
   --keep-last-epochs 1 --required-batch-size-multiple 1 \
   --save-dir ${SAVE} --log-format simple --log-interval 1000 --num-workers 0 | tee -a ${SAVE_LOG}/${exp_name}_log.txt

python /workspace/Bi-RAM/fairseq_cli/validate.py ${DATA} --task lra-text --batch-size 64 --valid-subset test --path ${SAVE}/checkpoint_best.pt | tee -a ${SAVE_LOG}/${exp_name}_log.txt
