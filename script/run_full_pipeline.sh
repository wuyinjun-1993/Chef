#!/bin/bash



python_cmd=python3


cd ../iterative_detect/


#batch_size=5000
batch_size=$8

#epochs=50
epochs=$7


#lr=0.0005
lr=$5

#l2_decay=0.05
l2_decay=$6


#dataset_name=retina

dataset_name=$2


model=Logistic_regression


#remove_count=50

remove_count=$3

dataset_dir=$1



#repeat_num=10

repeat_num=$4


clear_hist=$9

running_iteration=${10}

isGPU=${11}

GPUID=${12}


clear_hist_str=''

if [ "${clear_hist}" = true ];
then
	clear_hist_str='--restart'
fi


echo "restart or not::" $clear_hist_str


echo "${python_cmd} full_pipeline.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --start ${clear_hist_str} > ${dataset_dir}/output_start_${running_iteration}.txt 2>&1"


${python_cmd} full_pipeline.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --start ${clear_hist_str} > ${dataset_dir}/output_start_${running_iteration}.txt 2>&1



for (( i = 0 ; i < ${repeat_num} ; i++ ))
do

	echo "iteration $i"

	if [ "${isGPU}" = true ];
	then

		echo "${python_cmd} full_pipeline.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} > ${dataset_dir}/output_${i}_${running_iteration}.txt 2>&1"


		${python_cmd} full_pipeline.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} > ${dataset_dir}/output_${i}_${running_iteration}.txt 2>&1

<<cmd
		echo "${python_cmd} full_pipeline.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} --incremental > ${dataset_dir}/output_${i}_incremental_${running_iteration}.txt 2>&1"


		${python_cmd} full_pipeline.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} --incremental > ${dataset_dir}/output_${i}_incremental_${running_iteration}.txt 2>&1
cmd
	else

		echo "${python_cmd} full_pipeline.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} > ${dataset_dir}/output_${i}_${running_iteration}.txt 2>&1"

		${python_cmd} full_pipeline.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} > ${dataset_dir}/output_${i}_${running_iteration}.txt 2>&1

<<cmd
		echo "${python_cmd} full_pipeline.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --incremental > ${dataset_dir}/output_${i}_incremental_${running_iteration}.txt 2>&1"

		${python_cmd} full_pipeline.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --incremental > ${dataset_dir}/output_${i}_incremental_${running_iteration}.txt 2>&1
cmd

	fi




done

cd ../script/

