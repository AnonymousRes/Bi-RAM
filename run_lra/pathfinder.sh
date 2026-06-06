#! /bin/bash


DATA=/workspace/lra/pathfinder
SAVE_ROOT=/workspace/lra_saved_biram/pathfinder
exp_name=pathfinder_biram
SAVE=${SAVE_ROOT}/${exp_name}
SAVE_LOG=/workspace/Bi-RAM/out_log
rm -rf ${SAVE}
rm -f ${SAVE_LOG}/${exp_name}_log.txt
mkdir -p ${SAVE}

model=biram_lra_pf32
export CUDA_VISIBLE_DEVICES=1

python -u /workspace/Bi-RAM/train.py ${DATA} \
   --seed 2026 \
   --distributed-world-size 1 --ddp-backend c10d --find-unused-parameters \
   -a ${model} --task lra-image --input-type image --pixel-normalization 0.5 0.5 \
   --encoder-layers 6 --encoder-attention-heads 8 --encoder-embed-dim 128 --encoder-ffn-embed-dim 128 \
   --activation-fn 'silu' --attention-activation-fn 'softmax' \
   --norm-type 'batchnorm' --sen-rep-type 'mp' --encoder-normalize-before \
   --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
   --optimizer adam --lr 0.0008 --adam-betas '(0.9, 0.99)' --adam-eps 1e-8 --clip-mode='total' --clip-norm 1.0 \
   --dropout 0.3 --act-dropout 0.3 --weight-decay 0.05 \
   --batch-size 360 --sentence-avg --update-freq 1 \
   --lr-scheduler 'fixed' --max-epoch 400 \
   --keep-last-epochs 1 --max-sentences-valid 64 \
   --save-dir ${SAVE} --log-format simple --log-interval 300 --num-workers 0 | tee -a ${SAVE_LOG}/${exp_name}_log.txt

python /workspace/Bi-RAM/fairseq_cli/validate.py ${DATA} --task lra-image --batch-size 64 --valid-subset test --path ${SAVE}/checkpoint_best.pt | tee -a ${SAVE_LOG}/${exp_name}_log.txt
