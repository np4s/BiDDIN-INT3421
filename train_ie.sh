#modal: tva tv ta av t v a

#!/usr/bin/env bash
# export CUDA_VISIBLE_DEVICES=3   

EXP_NO="t2"
MODALS="tva"

echo "IEMOCAP, ${MODALS}, ${EXP_NO}"

LOG_PATH="./logs/IEMOCAP/${MODALS}"
if [[ ! -d ${LOG_PATH} ]];then
    mkdir -p  ${LOG_PATH}
fi

python -u train.py \
--dataset "IEMOCAP" \
--data_dir "./data/IEMOCAP_features/IEMOCAP_features_raw.pkl" \
--name ${EXP_NO} \
--modals ${MODALS} \
--class-weight \
--log_dir ${LOG_PATH}/${EXP_NO} \
--beta 0.5 \
--gamma 0.001 \
--lr 0.0001 \
--modulation \
--dropout 0.1 \
--tau 1 \
--epochs 40 \
>> ${LOG_PATH}/${EXP_NO}.out 2>&1
