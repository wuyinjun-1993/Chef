#!/bin/bash



python_cmd=python3


cd ../iterative_detect/


batch_size=5000

epochs=50


lr=0.0005


l2_decay=0.01


dataset_name=retina


model=Logistic_regression


remove_count=50

dataset_dir=$1



repeat_num=50

o2u_epoch=$2

isGPU=$3

GPUID=$4





echo "${python_cmd} full_pipeline_o2u.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --start --o2u_epochs ${o2u_epoch} > ${dataset_dir}/output_o2u_start.txt 2>&1"


${python_cmd} full_pipeline_o2u.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --start --o2u_epochs ${o2u_epoch} > ${dataset_dir}/output_o2u_start.txt 2>&1



for (( i = 0 ; i < ${repeat_num} ; i++ ))
do

	echo "iteration $i"

	if [ "${isGPU}" = true ];
	then
	
		echo "${python_cmd} full_pipeline_o2u.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} --o2u_epochs ${o2u_epoch} > ${dataset_dir}/output_o2u_${i}.txt 2>&1"

		${python_cmd} full_pipeline_o2u.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} --o2u_epochs ${o2u_epoch} > ${dataset_dir}/output_o2u_${i}.txt 2>&1


#		${python_cmd} full_pipeline_o2u.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} --incremental > ${dataset_dir}/output_o2u_${i}_incremental.txt 2>&1

	else

		echo "${python_cmd} full_pipeline_o2u.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --o2u_epochs ${o2u_epoch} > ${dataset_dir}/output_o2u_${i}.txt 2>&1"

		${python_cmd} full_pipeline_o2u.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --o2u_epochs ${o2u_epoch} > ${dataset_dir}/output_o2u_${i}.txt 2>&1

#		${python_cmd} full_pipeline.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --incremental > ${dataset_dir}/output_${i}_incremental.txt 2>&1


	fi




done

