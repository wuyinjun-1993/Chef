#!/bin/bash



python_cmd=python3


cd ../iterative_detect/


batch_size=$8

epochs=$7

lr=$5


l2_decay=$6


#outer_loop_count=2000


dataset_name=$2

model=Logistic_regression


remove_count=$3

dataset_dir=$1




repeat_num=$4

#duti_training_lr=200

clear_hist=$9


running_iteration=${10}

isGPU=${11}

GPUID=${12}


tag=${13}

clear_hist_str=''

GPU_str=''

if [ "${isGPU}" = true ];
then
	GPU_str="--GPU --GPUID ${GPUID}"

fi

#source ../script/hyper_${dataset_name}_one

source ../script/hyper_${dataset_name}_one${tag}

echo "lr::"$lr
echo "l2 decay::" ${l2_decay}
echo "epochs::" ${epochs}
echo "batch size::" ${batch_size}
echo "duti_training_lr::"${duti_training_lr}
echo "outer_loop_count::"${outer_loop_count}


echo "${python_cmd} Duti_python.py --lr ${duti_training_lr} --bz ${batch_size} --epochs ${epochs}  --out_epoch_count ${outer_loop_count} --inner_epoch_count 200 --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name}  --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} ${GPU_str}> ${dataset_dir}/output_duti_${running_iteration}.txt 2>&1"

${python_cmd} Duti_python.py --lr ${duti_training_lr} --bz ${batch_size} --epochs ${epochs}  --out_epoch_count ${outer_loop_count} --inner_epoch_count 200 --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name}  --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} ${GPU_str}> ${dataset_dir}/output_duti_${running_iteration}.txt 2>&1



cd ../script/

