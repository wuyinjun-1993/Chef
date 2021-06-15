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

derive_lr=${14}

regular_rate=${15}

regular_str="--regular_rate ${regular_rate}"


clear_hist_str=''

if [ "${clear_hist}" = true ];
then
	clear_hist_str='--restart'
fi


echo "restart or not::" $clear_hist_str


hist_period=$1

hist_period_str="--hist_period ${hist_period}"


#echo "${python_cmd} full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --start ${clear_hist_str} ${hist_period_str} > ${dataset_dir}/output_vary_remove_count_start_sl_${running_iteration}.txt 2>&1"


#${python_cmd} full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --start ${clear_hist_str} ${hist_period_str}  > ${dataset_dir}/output_vary_remove_count_start_sl_${running_iteration}.txt 2>&1


repeat_sample_count_list=(6000 400 30)

repeat_times_list=(1 15 200)

continue_labeling_str='--continue_labeling'

for k in "${!repeat_sample_count_list[@]}"
do
	echo "repeat times::" $k ${repeat_sample_count_list[$k]} ${repeat_times_list[$k]}


	repeat_num=${repeat_times_list[$k]}

	remove_count=${repeat_sample_count_list[$k]}

	if [ "$k" -gt 0 ];
	then
		clear_hist_str=''
	fi

	if [ "${isGPU}" = true ];
	then

		echo "${python_cmd} full_pipeline_suggest_labels.py --lr ${derive_lr} --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count 50 --output_dir ${dataset_dir} --start ${clear_hist_str} ${hist_period_str} --GPU --GPUID ${GPUID} > ${dataset_dir}/output_vary_remove_count_${remove_count}_start_sl_${running_iteration}.txt 2>&1"


		${python_cmd} full_pipeline_suggest_labels.py --lr ${derive_lr} --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count 50 --output_dir ${dataset_dir} --start ${clear_hist_str} ${hist_period_str} --GPU --GPUID ${GPUID}  > ${dataset_dir}/output_vary_remove_count_${remove_count}_start_sl_${running_iteration}.txt 2>&1
	else
		echo "${python_cmd} full_pipeline_suggest_labels.py --lr ${derive_lr} --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count 50 --output_dir ${dataset_dir} --start ${clear_hist_str} ${hist_period_str} > ${dataset_dir}/output_vary_remove_count_${remove_count}_start_sl_${running_iteration}.txt 2>&1"


                ${python_cmd} full_pipeline_suggest_labels.py --lr ${derive_lr} --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count 50 --output_dir ${dataset_dir} --start ${clear_hist_str} ${hist_period_str}  > ${dataset_dir}/output_vary_remove_count_${remove_count}_start_sl_${running_iteration}.txt 2>&1

	fi

	for (( i = 0 ; i < ${repeat_num} ; i++ ))
	do

		echo "iteration $i"

		if [ "${isGPU}" = true ];
		then


#			echo "${python_cmd} full_pipeline_suggest_labels.py --lr ${derive_lr} --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} ${hist_period_str} --resolve_conflict 0  > ${dataset_dir}/output_resolve_conflict_0_remove_count_${remove_count}_${i}_sl_${running_iteration}.txt 2>&1"


 #                       ${python_cmd} full_pipeline_suggest_labels.py --lr ${derive_lr} --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} ${hist_period_str} --resolve_conflict 0  > ${dataset_dir}/output_resolve_conflict_0_remove_count_${remove_count}_${i}_sl_${running_iteration}.txt 2>&1


#			echo "${python_cmd} full_pipeline_suggest_labels.py --lr ${derive_lr} --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} ${hist_period_str} --resolve_conflict 1  > ${dataset_dir}/output_resolve_conflict_1_remove_count_${remove_count}_${i}_sl_${running_iteration}.txt 2>&1"


 #                       ${python_cmd} full_pipeline_suggest_labels.py --lr ${derive_lr} --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} ${hist_period_str} --resolve_conflict 1  > ${dataset_dir}/output_resolve_conflict_1_remove_count_${remove_count}_${i}_sl_${running_iteration}.txt 2>&1




			echo "${python_cmd} full_pipeline_suggest_labels.py --lr ${derive_lr} --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} ${hist_period_str} --resolve_conflict 3 ${continue_labeling_str} ${regular_str} > ${dataset_dir}/output_resolve_conflict_3_remove_count_${remove_count}_${i}_sl_${running_iteration}.txt 2>&1"


                        ${python_cmd} full_pipeline_suggest_labels.py --lr ${derive_lr} --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} ${hist_period_str} --resolve_conflict 3 ${continue_labeling_str} ${regular_str}  > ${dataset_dir}/output_resolve_conflict_3_remove_count_${remove_count}_${i}_sl_${running_iteration}.txt 2>&1




#			echo "${python_cmd} full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} ${hist_period_str} ${continue_labeling_str}  > ${dataset_dir}/output_remove_count_${remove_count}_${i}_sl_${running_iteration}.txt 2>&1"


#			${python_cmd} full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} ${hist_period_str} ${continue_labeling_str}  > ${dataset_dir}/output_remove_count_${remove_count}_${i}_sl_${running_iteration}.txt 2>&1


	#		echo "${python_cmd} full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} --incremental ${hist_period_str} > ${dataset_dir}/output_remove_count_${remove_count}_${i}_incremental_sl_${running_iteration}.txt 2>&1"


	#		${python_cmd} full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --GPU --GPUID ${GPUID} --incremental ${hist_period_str}  > ${dataset_dir}/output_remove_count_${remove_count}_${i}_incremental_sl_${running_iteration}.txt 2>&1

		else

#			echo "${python_cmd} full_pipeline_suggest_labels.py --lr ${derive_lr} --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir}  ${hist_period_str} --resolve_conflict 0  > ${dataset_dir}/output_resolve_conflict_0_remove_count_${remove_count}_${i}_sl_${running_iteration}.txt 2>&1"


 #                      ${python_cmd} full_pipeline_suggest_labels.py --lr ${derive_lr} --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} ${hist_period_str} --resolve_conflict 0 > ${dataset_dir}/output_resolve_conflict_0_remove_count_${remove_count}_${i}_sl_${running_iteration}.txt 2>&1


#			echo "${python_cmd} full_pipeline_suggest_labels.py --lr ${derive_lr} --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir}  ${hist_period_str} --resolve_conflict 1 > ${dataset_dir}/output_resolve_conflict_1_remove_count_${remove_count}_${i}_sl_${running_iteration}.txt 2>&1"


 #                      ${python_cmd} full_pipeline_suggest_labels.py --lr ${derive_lr} --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} ${hist_period_str} --resolve_conflict 1 > ${dataset_dir}/output_resolve_conflict_1_remove_count_${remove_count}_${i}_sl_${running_iteration}.txt 2>&1



			echo "${python_cmd} full_pipeline_suggest_labels.py --lr ${derive_lr} --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} ${hist_period_str} --resolve_conflict 3 ${continue_labeling_str} ${regular_str} > ${dataset_dir}/output_resolve_conflict_3_remove_count_${remove_count}_${i}_sl_${running_iteration}.txt 2>&1"


                        ${python_cmd} full_pipeline_suggest_labels.py --lr ${derive_lr} --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir}  ${hist_period_str} --resolve_conflict 3 ${continue_labeling_str} ${regular_str} > ${dataset_dir}/output_resolve_conflict_3_remove_count_${remove_count}_${i}_sl_${running_iteration}.txt 2>&1


#			echo "${python_cmd} full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} ${hist_period_str} ${continue_labeling_str}  > ${dataset_dir}/output_remove_count_${remove_count}_${i}_sl_${running_iteration}.txt 2>&1"

#			${python_cmd} full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} ${hist_period_str} ${continue_labeling_str}  > ${dataset_dir}/output_remove_count_${remove_count}_${i}_sl_${running_iteration}.txt 2>&1


	#		echo "${python_cmd} full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --incremental ${hist_period_str} > ${dataset_dir}/output_remove_count_${remove_count}_${i}_incremental_sl_${running_iteration}.txt 2>&1"

	#		${python_cmd} full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --incremental ${hist_period_str} > ${dataset_dir}/output_remove_count_${remove_count}_${i}_incremental_sl_${running_iteration}.txt 2>&1


		fi



		removed_iter_id=$(( $i - 5 ))

                if [[ ${removed_iter_id} -gt 0 ]]
                then

                        echo "${dataset_dir}/full_training_noisy_dataset_sl_v${removed_iter_id}*"

                        rm ${dataset_dir}/full_training_noisy_dataset_sl_v${removed_iter_id}*

                        rm ${dataset_dir}/Logistic_regression_influence_v${removed_iter_id}*

                        rm ${dataset_dir}/full_existing_labeled_id_tensor_sl_v${removed_iter_id}*

                fi


	done
done

cd ../script/

