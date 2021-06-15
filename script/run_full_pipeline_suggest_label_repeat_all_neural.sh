#!/bin/bash



python_cmd=python3



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

source_file=${20}

source ${source_file}


echo "source file:: ${source_file}"

model=${model}


#remove_count=50

remove_count=$4

dataset_dir=$2



#repeat_num=10

repeat_num=$5


clear_hist=${10}

running_iteration=${11}

isGPU=${12}

GPUID=${13}

derive_lr=${derived_lr}

regular_rate=${15}

python_file=${16}

multi_strategy=${17}

suffix=${18}

regular_str="--regular_rate ${regular_rate}"


default_strategy=${19}

curr_y_diff=${22}


clear_hist_str=''


curr_y_dff_str=''

if [ "${clear_hist}" = true ];
then
	clear_hist_str='--restart'
fi

if [ "${curr_y_diff}" = true ];
then
        curr_y_dff_str='--same_y_diff'
fi



suffix_str="--suffix ${suffix}"


echo "restart or not::" $clear_hist_str

cd ../iterative_detect/

hist_period=$1

hist_period_str="--hist_period ${hist_period}"

echo "${python_cmd} gen_simulated_labels.py --dataset ${dataset_name} --output_dir ${dataset_dir}"

${python_cmd} gen_simulated_labels.py --dataset ${dataset_name} --output_dir ${dataset_dir}

#echo "${python_cmd} full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --start ${clear_hist_str} ${hist_period_str} > ${dataset_dir}/output_vary_remove_count_start_sl_${running_iteration}.txt 2>&1"


#${python_cmd} full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${remove_count} --output_dir ${dataset_dir} --start ${clear_hist_str} ${hist_period_str}  > ${dataset_dir}/output_vary_remove_count_start_sl_${running_iteration}.txt 2>&1

#source ./hyper_${dataset_name}


repeat_sample_count_list=${repeat_sample_count_list} #(6000 600 100 60)

#repeat_times_list=${repeat_times_list} #(1 10 60 100)

remove_total_count=${removed_total_count}



continue_labeling_str='--continue_labeling'

for k in "${!repeat_sample_count_list[@]}"
do
	echo "repeat times::" $k ${repeat_sample_count_list[$k]} ${repeat_times_list[$k]}


#	repeat_num=${repeat_times_list[$k]}

	remove_count=${repeat_sample_count_list[$k]}

	if [ "$k" -gt 0 ];
	then
		clear_hist_str=''


	fi




	if [ "${isGPU}" = true ];
	then

		echo "${python_cmd} ${python_file}.py --derived_lr ${derive_lr} --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count 50 --output_dir ${dataset_dir} --start ${clear_hist_str} ${hist_period_str} --GPU --GPUID ${GPUID} ${regular_str} ${suffix_str} > ${dataset_dir}/output_vary_remove_count_${remove_count}_start_${suffix}_${running_iteration}.txt 2>&1"


		${python_cmd} ${python_file}.py --derived_lr ${derive_lr} --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count 50 --output_dir ${dataset_dir} --start ${clear_hist_str} ${hist_period_str} --GPU --GPUID ${GPUID} ${regular_str} ${suffix_str} > ${dataset_dir}/output_vary_remove_count_${remove_count}_start_${suffix}_${running_iteration}.txt 2>&1
	else
		echo "${python_cmd} ${python_file}.py --derived_lr ${derive_lr} --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count 50 --output_dir ${dataset_dir} --start ${clear_hist_str} ${hist_period_str} ${regular_str} ${suffix_str} > ${dataset_dir}/output_vary_remove_count_${remove_count}_start_${suffix}_${running_iteration}.txt 2>&1"


                ${python_cmd} ${python_file}.py --derived_lr ${derive_lr} --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count 50 --output_dir ${dataset_dir} --start ${clear_hist_str} ${hist_period_str} ${regular_str} ${suffix_str}  > ${dataset_dir}/output_vary_remove_count_${remove_count}_start_${suffix}_${running_iteration}.txt 2>&1

	fi


done

rm ${dataset_dir}/full_training_noisy_dataset_${suffix}*

rm ${dataset_dir}/Logistic_regression_influence_${suffix}*

rm ${dataset_dir}/full_existing_labeled_id_tensor_${suffix}*

rm ${dataset_dir}/w_list_${suffix}*

rm ${dataset_dir}/grad_list_${suffix}*


#rm ${dataset_dir}/*${suffix}*

cd ../script/

