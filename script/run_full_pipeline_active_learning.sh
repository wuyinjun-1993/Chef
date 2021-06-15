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



repeat_num=20



isGPU=$2

GPUID=$3


echo "${python_cmd} full_pipeline_active_learning.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --start > ${dataset_dir}/output_start_ac.txt 2>&1"


${python_cmd} full_pipeline_active_learning.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --start > ${dataset_dir}/output_start_ac.txt 2>&1



for (( i = 0 ; i < ${repeat_num} ; i++ ))
do

	echo "iteration $i"

	if [ "${isGPU}" = true ];
	then

		echo "${python_cmd} full_pipeline_active_learning.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} > ${dataset_dir}/output_${i}_ac.txt 2>&1"


		${python_cmd} full_pipeline_active_learning.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} > ${dataset_dir}/output_${i}_ac.txt 2>&1


#		echo "${python_cmd} full_pipeline_active_learning.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} --incremental > ${dataset_dir}/output_${i}_incremental_ac.txt 2>&1"


#		${python_cmd} full_pipeline_active_learning.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} --incremental > ${dataset_dir}/output_${i}_incremental_ac.txt 2>&1

	else

		echo "${python_cmd} full_pipeline_active_learning.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} > ${dataset_dir}/output_${i}_ac.txt 2>&1"

		${python_cmd} full_pipeline_active_learning.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} > ${dataset_dir}/output_${i}_ac.txt 2>&1


#		echo "${python_cmd} full_pipeline_active_learning.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --incremental > ${dataset_dir}/output_${i}_incremental_ac.txt 2>&1"

#		${python_cmd} full_pipeline_active_learning.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --incremental > ${dataset_dir}/output_${i}_incremental_ac.txt 2>&1


	fi




done

