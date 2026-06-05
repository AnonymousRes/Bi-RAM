#! /bin/bash


DATA=/workspace/lra/cifar10
SAVE_ROOT=/workspace/lra_saved_biram/cifar10
exp_name=image_biram
SAVE=${SAVE_ROOT}/${exp_name}
SAVE_LOG=/workspace/Bi-RAM/out_log
#rm -rf ${SAVE}
#rm -f ${SAVE_LOG}/${exp_name}_log.txt
#mkdir -p ${SAVE}

model=biram_lra_cifar10
export CUDA_VISIBLE_DEVICES=1

#python -u /workspace/Bi-RAM/train.py ${DATA} \
#    --seed 2026 \
#    --distributed-world-size 1 --ddp-backend c10d --find-unused-parameters \
#    -a ${model} --task lra-image --input-type image --pixel-normalization 0.48 0.24 \
#    --encoder-layers 8 --encoder-attention-heads 8 --encoder-embed-dim 160 --encoder-ffn-embed-dim 160 \
#    --activation-fn 'silu' --attention-activation-fn 'softmax' \
#    --norm-type 'batchnorm' --sen-rep-type 'mp' --encoder-normalize-before \
#    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
#    --optimizer adam --lr 0.0006 --adam-betas '(0.9, 0.99)' --adam-eps 1e-8 --clip-norm 1.0 \
#    --dropout 0.3 --act-dropout 0.3 --weight-decay 0.05 \
#    --batch-size 300 --sentence-avg --update-freq 1 \
#    --lr-scheduler linear_decay --end-learning-rate 0.0 --max-update 60000 --warmup-updates 1500 --warmup-init-lr '1e-07' \
#    --keep-last-epochs 1 --required-batch-size-multiple 1 \
#    --save-dir ${SAVE} --log-format simple --log-interval 500 --num-workers 0 | tee -a ${SAVE_LOG}/${exp_name}_log.txt
python /workspace/Bi-RAM/fairseq_cli/validate.py ${DATA} --task lra-image --batch-size 64 --valid-subset test --path ${SAVE}/checkpoint_best.pt | tee -a ${SAVE_LOG}/${exp_name}_log.txt
#--lr-scheduler fixed --max-epoch 200 \
#    --lr-scheduler fixed --max-epoch 200 \
#--lr-scheduler linear_decay --total-num-update 25000 --max-update 25000 --warmup-updates 1500 --warmup-init-lr '1e-07' \
#--lr-scheduler linear_decay --total-num-update 360000 --end-learning-rate 0.0 --max-update 360000 --warmup-updates 25000 --warmup-init-lr '1e-07' \
#--lr-scheduler linear_decay --total-num-update 100000 --end-learning-rate 0.0 --max-update 100000 --warmup-updates 8000 --warmup-init-lr '1e-07' \
#DATA=~/lra_data/cifar10
#SAVE_ROOT=~/saved_xformer/cifar10
#exp_name=image_biram
#SAVE=${SAVE_ROOT}/${exp_name}
#SAVE_LOG=~/projects/Lee_HP1203/out_log
#rm -rf ${SAVE}
#rm -f ${SAVE_LOG}/${exp_name}_log.txt
#mkdir -p ${SAVE}
#
#model=biram_lra_cifar10
#export CUDA_VISIBLE_DEVICES=1
#
#python -u ~/projects/Lee_HP1203/train.py ${DATA} \
#    --seed 2025 \
#    --distributed-world-size 1 --ddp-backend c10d --find-unused-parameters \
#    -a ${model} --task lra-image --input-type image --pixel-normalization 0.5 0.5 \
#    --encoder-layers 8 --encoder-attention-heads 8 --encoder-embed-dim 160 --encoder-ffn-embed-dim 160 \
#    --activation-fn 'silu' --attention-activation-fn 'softmax' \
#    --norm-type 'batchnorm' --sen-rep-type 'mp' --encoder-normalize-before \
#    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
#    --optimizer adam --lr 0.0008 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
#    --dropout 0.2 --attention-dropout 0.2 --act-dropout 0.2 --weight-decay 0.10 \
#    --batch-size 200 --max-epoch 200 --sentence-avg --update-freq 1 --max-update 99999999 \
#    --lr-scheduler fixed \
#    --warmup-updates 500 --keep-last-epochs 1 --required-batch-size-multiple 1 \
#    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 | tee -a ${SAVE_LOG}/${exp_name}_log.txt
#date
#python ~/projects/Lee_HP1203/fairseq_cli/validate.py ${DATA} --task lra-image --batch-size 64 --valid-subset test --path ${SAVE}/checkpoint_best.pt | tee -a ${SAVE_LOG}/${exp_name}_log.txt
##--lr-scheduler fixed --total-num-update 9999999  --end-learning-rate 0.0  --warmup-init-lr '1e-07' \