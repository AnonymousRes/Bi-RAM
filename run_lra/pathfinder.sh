#! /bin/bash


DATA=/workspace/lra/pathfinder
SAVE_ROOT=/workspace/lra_saved_biram/pathfinder
exp_name=pathfinder_biram
SAVE=${SAVE_ROOT}/${exp_name}
SAVE_LOG=/workspace/Bi-RAM/out_log
#rm -rf ${SAVE}
#rm -f ${SAVE_LOG}/${exp_name}_log.txt
#mkdir -p ${SAVE}

model=biram_lra_pf32
export CUDA_VISIBLE_DEVICES=1

#python -u /workspace/Bi-RAM/train.py ${DATA} \
#    --seed 2026 \
#    --distributed-world-size 1 --ddp-backend c10d --find-unused-parameters \
#    -a ${model} --task lra-image --input-type image --pixel-normalization 0.5 0.5 \
#    --encoder-layers 6 --encoder-attention-heads 8 --encoder-embed-dim 128 --encoder-ffn-embed-dim 128 \
#    --activation-fn 'silu' --attention-activation-fn 'softmax' \
#    --norm-type 'batchnorm' --sen-rep-type 'mp' --encoder-normalize-before \
#    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
#    --optimizer adam --lr 0.0008 --adam-betas '(0.9, 0.99)' --adam-eps 1e-8 --clip-mode='total' --clip-norm 1.0 \
#    --dropout 0.3 --act-dropout 0.3 --weight-decay 0.05 \
#    --batch-size 360 --sentence-avg --update-freq 1 \
#    --lr-scheduler 'fixed' --max-epoch 400 \
#    --keep-last-epochs 1 --max-sentences-valid 64 \
#    --save-dir ${SAVE} --log-format simple --log-interval 300 --num-workers 0 | tee -a ${SAVE_LOG}/${exp_name}_log.txt

#    --lr-scheduler 'linear_decay' --total-num-update 250000 --max-update 250000 --end-learning-rate 0.0 --warmup-updates 50000 --warmup-init-lr '1e-07' \
#        --lr-scheduler 'fixed' --max-epoch 100 \
#--lr-scheduler 'linear_decay' --total-num-update 250000 --end-learning-rate 0.0 --warmup-updates 50000 --warmup-init-lr '1e-07' \
#    --lr-scheduler 'linear_decay' --total-num-update 250000 --max-update 250000 --end-learning-rate 0.0 --warmup-updates 50000 --warmup-init-lr '1e-07' \
python /workspace/Bi-RAM/fairseq_cli/validate.py ${DATA} --task lra-image --batch-size 64 --valid-subset test --path ${SAVE}/checkpoint_best.pt | tee -a ${SAVE_LOG}/${exp_name}_log.txt

#    --lr-scheduler 'linear_decay' --total-num-update 80000 --max-update 80000 --end-learning-rate 0.0 --warmup-updates 12000 --warmup-init-lr '1e-07' \
#DATA=~/lra_data/pathfinder
#SAVE_ROOT=~/saved_xformer/pathfinder
#exp_name=pathfinder_biram
#SAVE=${SAVE_ROOT}/${exp_name}
#SAVE_LOG=~/projects/Lee_HP1203/out_log
#rm -rf ${SAVE}
#rm -f ${SAVE_LOG}/${exp_name}_log.txt
#mkdir -p ${SAVE}

#model=biram_lra_pf32
#export CUDA_VISIBLE_DEVICES=0
#
#python -u ~/projects/Lee_HP1203/train.py ${DATA} \
#    --seed 2025 \
#    --distributed-world-size 1 --ddp-backend c10d --find-unused-parameters \
#    -a ${model} --task lra-image --input-type image --pixel-normalization 0.5 0.5 \
#    --encoder-layers 4 --encoder-attention-heads 8 --encoder-embed-dim 128 --encoder-ffn-embed-dim 128 \
#    --activation-fn 'silu' --attention-activation-fn 'softmax' \
#    --norm-type 'batchnorm' --sen-rep-type 'mp' --encoder-normalize-before \
#    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
#    --optimizer adam --lr 0.0003 --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-mode='total' --clip-norm 1.0 \
#    --dropout 0.15 --weight-decay 0.15 \
#    --batch-size 128 --sentence-avg --update-freq 1 \
#    --lr-scheduler 'fixed' --max-epoch 200 \
#    --keep-last-epochs 1 --max-sentences-valid 1 \
#    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 | tee -a ${SAVE_LOG}/${exp_name}_log.txt
#
#python ~/projects/Lee_HP1203/fairseq_cli/validate.py ${DATA} --task lra-image --batch-size 64 --valid-subset test --path ${SAVE}/checkpoint_best.pt | tee -a ${SAVE_LOG}/${exp_name}_log.txt