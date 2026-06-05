#! /bin/bash


DATA=/workspace/lra/aan
SAVE_ROOT=/workspace/lra_saved_biram/aan
exp_name=aan_biram
SAVE=${SAVE_ROOT}/${exp_name}
SAVE_LOG=/workspace/Bi-RAM/out_log
#rm -rf ${SAVE}
#rm -f ${SAVE_LOG}/${exp_name}_log.txt
#mkdir -p ${SAVE}

model=biram_lra_aan
export CUDA_VISIBLE_DEVICES=1

#python -u /workspace/Bi-RAM/train.py ${DATA} \
#    --seed 2026 \
#    --distributed-world-size 1 --ddp-backend c10d --find-unused-parameters \
#    -a ${model} --task lra-text --input-type text \
#    --encoder-layers 6 --encoder-attention-heads 8 --encoder-embed-dim 128 --encoder-ffn-embed-dim 128 \
#    --activation-fn 'silu' --attention-activation-fn 'softmax' \
#    --norm-type 'scalenorm' --sen-rep-type 'mp' --encoder-normalize-before \
#    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
#    --optimizer adam --lr 0.0015 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
#    --dropout 0.2 --act-dropout 0.2 --weight-decay 0.02 \
#    --batch-size 35 --sentence-avg --update-freq 1 \
#    --lr-scheduler linear_decay --total-num-update 300000 --max-update 300000 --end-learning-rate 0.0 --warmup-updates 30000 --warmup-init-lr '1e-07' \
#    --keep-last-epochs 1 --max-sentences-valid 16 \
#    --save-dir ${SAVE} --log-format simple --log-interval 10000 --num-workers 0 | tee -a ${SAVE_LOG}/${exp_name}_log.txt

python /workspace/Bi-RAM/fairseq_cli/validate.py ${DATA} --task lra-text --batch-size 32 --valid-subset test --path ${SAVE}/checkpoint_best.pt | tee -a ${SAVE_LOG}/${exp_name}_log.txt
#python /workspace/Bi-RAM/fairseq_cli/validate.py ${DATA} --task lra-text --batch-size 32 --valid-subset test --path ${SAVE}/checkpoint_best.pt




#DATA=~/lra_data/aan
#SAVE_ROOT=~/saved_xformer/aan
#exp_name=aan_biram
#SAVE=${SAVE_ROOT}/${exp_name}
#SAVE_LOG=~/projects/Lee_HP1203/out_log
#rm -rf ${SAVE}
#rm -f ${SAVE_LOG}/${exp_name}_log.txt
#mkdir -p ${SAVE}
#
#model=biram_lra_aan
#export CUDA_VISIBLE_DEVICES=1
#
#python -u ~/projects/Lee_HP1203/train.py ${DATA} \
#    --seed 2025 \
#    --distributed-world-size 1 --ddp-backend c10d --find-unused-parameters \
#    -a ${model} --task lra-text --input-type text \
#    --encoder-layers 6 --encoder-attention-heads 8 --encoder-embed-dim 128 --encoder-ffn-embed-dim 128 \
#    --activation-fn 'silu' --attention-activation-fn 'softmax' \
#    --norm-type 'scalenorm' --sen-rep-type 'mp' \
#    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
#    --optimizer adam --lr 0.0008 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
#    --dropout 0.1 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.04 \
#    --batch-size 32 --max-epoch 100 --sentence-avg --update-freq 1 \
#    --lr-scheduler 'fixed' \
#    --keep-last-epochs 1 --max-sentences-valid 128 \
#    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 | tee -a ${SAVE_LOG}/${exp_name}_log.txt
#
#python ~/projects/Lee_HP1203/fairseq_cli/validate.py ${DATA} --task lra-text --batch-size 20 --valid-subset test --path ${SAVE}/checkpoint_best.pt | tee -a ${SAVE_LOG}/${exp_name}_log.txt
#2026-01-23 19:05:29 | INFO | test |  | valid on 'test' subset | loss 0.422 | accuracy 88.805 | wps 0 | wpb 6.97654e+07 | bsz 17437