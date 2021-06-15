#!/bin/bash



python_cmd=python3


cd ../iterative_detect/


#batch_size=5000
batch_size=$9

#epochs=50
epochs=$8


#lr=0.0005
lr=$6

#l2_decay=0.05
l2_decay=$7


#dataset_name=retina

dataset_name=$3


model=Logistic_regression


#remove_count=50

remove_count=$4

dataset_dir=$2



#repeat_num=10

repeat_num=$5


clear_hist=${10}

running_iteration=${11}

isGPU=${12}

GPUID=${13}


clear_hist_str=''

if [ "${clear_hist}" = true ];
then
	clear_hist_str='--restart'
fi


echo "restart or not::" $clear_hist_str

gpu_measure='--GPU_measure'


hist_period=$1

hist_period_str="--hist_period ${hist_period}"


echo "${python_cmd} full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --start ${clear_hist_str} ${hist_period_str} > ${dataset_dir}/output_start_sl_gpu_${running_iteration}.txt 2>&1"


${python_cmd} full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --start ${clear_hist_str} ${hist_period_str}  > ${dataset_dir}/output_start_sl_gpu_${running_iteration}.txt 2>&1



for (( i = 0 ; i < ${repeat_num} ; i++ ))
do

	echo "iteration $i"

	if [ "${isGPU}" = true ];
	then

		echo "${python_cmd} full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} ${hist_period_str} ${gpu_measure}  > ${dataset_dir}/output_${i}_sl_gpu_${running_iteration}.txt 2>&1"


		${python_cmd} full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} ${hist_period_str} ${gpu_measure}  > ${dataset_dir}/output_${i}_sl_gpu_${running_iteration}.txt 2>&1


		echo "${python_cmd} full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} --incremental ${hist_period_str} ${gpu_measure} > ${dataset_dir}/output_${i}_incremental_sl_gpu_${running_iteration}.txt 2>&1"


		${python_cmd} full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} --incremental ${hist_period_str} ${gpu_measure}  > ${dataset_dir}/output_${i}_incremental_sl_gpu_${running_iteration}.txt 2>&1

	else

		echo "${python_cmd} full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} ${hist_period_str} ${gpu_measure} > ${dataset_dir}/output_${i}_sl_gpu_${running_iteration}.txt 2>&1"

		${python_cmd} full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} ${hist_period_str} ${gpu_measure} > ${dataset_dir}/output_${i}_sl_gpu_${running_iteration}.txt 2>&1


		echo "${python_cmd} full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --incremental ${hist_period_str} ${gpu_measure} > ${dataset_dir}/output_${i}_incremental_sl_gpu_${running_iteration}.txt 2>&1"

		${python_cmd} full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --incremental ${hist_period_str} ${gpu_measure} > ${dataset_dir}/output_${i}_incremental_sl_gpu_${running_iteration}.txt 2>&1


	fi




done

cd ../script/

