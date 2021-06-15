#!/bin/bash



python_cmd=python3


cd ../iterative_detect/


batch_size=5000

epochs=50


lr=0.0005


l2_decay=0.05


dataset_name=retina


#model=Logistic_regression
model=Binary_Logistic_regression

remove_count=50

dataset_dir=$1



repeat_num=20

clear_hist=$2

isGPU=$3

GPUID=$4


clear_hist_str=''

if [ "${clear_hist}" = true ];
then
	clear_hist_str='--restart'
fi


echo "restart or not::" $clear_hist_str


echo "${python_cmd} full_pipeline_activeclean.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --start ${clear_hist_str} > ${dataset_dir}/output_ac_start.txt 2>&1"


${python_cmd} full_pipeline_activeclean.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --start ${clear_hist_str} > ${dataset_dir}/output_ac_start.txt 2>&1



for (( i = 0 ; i < ${repeat_num} ; i++ ))
do

	echo "iteration $i"

	if [ "${isGPU}" = true ];
	then

		echo "${python_cmd} full_pipeline_activeclean.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} > ${dataset_dir}/output_${i}_ac.txt 2>&1"


		${python_cmd} full_pipeline_activeclean.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} > ${dataset_dir}/output_${i}_ac.txt 2>&1


		echo "${python_cmd} full_pipeline_activeclean.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} --incremental > ${dataset_dir}/output_${i}_ac_incremental.txt 2>&1"


		${python_cmd} full_pipeline_activeclean.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} --incremental > ${dataset_dir}/output_${i}_ac_incremental.txt 2>&1

	else

		echo "${python_cmd} full_pipeline_activeclean.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} > ${dataset_dir}/output_${i}_ac.txt 2>&1"

		${python_cmd} full_pipeline_activeclean.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} > ${dataset_dir}/output_${i}_ac.txt 2>&1


		echo "${python_cmd} full_pipeline_activeclean.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --incremental > ${dataset_dir}/output_${i}_ac_incremental.txt 2>&1"

		${python_cmd} full_pipeline_activeclean.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --incremental > ${dataset_dir}/output_${i}_ac_incremental.txt 2>&1


	fi




done

