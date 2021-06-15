#!/bin/bash



python_cmd=python3


#cd ../iterative_detect/


#batch_size=$8

#epochs=$7

#lr=$5


#l2_decay=$6


#outer_loop_count=2000


dataset_name=$2

model=$5


remove_count=200

dataset_dir=$1




repeat_num=10

duti_training_lr=200

#clear_hist=$9

total_iteration=1

#running_iteration=${10}

isGPU=${3}

GPUID=${4}

source_file_suffix=$6

is_tars=false

if [ "$#" -ge 7 ]; then
    is_tars=${7}
fi

cd ../iterative_detect

#echo "${python_cmd} gen_simulated_labels.py --dataset ${dataset_name} --output_dir ${dataset_dir}"

#${python_cmd} gen_simulated_labels.py --dataset ${dataset_name} --output_dir ${dataset_dir}


if [ "${is_tars}" = true ];
then

        cd ../process_data

        echo "python3 process_dataset_for_tars.py --dataset ${dataset_name} --output_dir ${dataset_dir} > ${dataset_dir}/output_gen_tars_dataset.txt 2>&1"

        python3 process_dataset_for_tars.py --dataset ${dataset_name} --output_dir ${dataset_dir} > ${dataset_dir}/output_gen_tars_dataset.txt 2>&1


fi

cd ../script


echo "is_tars::${is_tars}"

echo "run DUTI"

source_file="./hyper_${dataset_name}${source_file_suffix}"

source ${source_file}

echo "lr::"$lr
echo "l2 decay::" ${l2_decay}
echo "epochs::" ${epochs}
echo "batch size::" ${batch_size}
echo "source file:: ${source_file}"

<<cmd
total_removed_count=$(( remove_count*repeat_num  ))


echo "total_removed_count::"${total_removed_count}

for (( i = 0 ; i < ${total_iteration} ; i++ ))
do
	echo "bash run_duti.sh ${1} $2 ${total_removed_count} 1 $lr ${l2_decay} ${epochs} ${batch_size} false ${i} ${isGPU} ${GPUID}"

	bash run_duti.sh ${1} $2 ${total_removed_count} 1 $lr ${l2_decay} ${epochs} ${batch_size} false ${i} ${isGPU} ${GPUID}

done
cmd


echo "start influence-l"
if [ "${dataset_name}" == 'retina' ];
then
	echo 'is retina';
else
	echo 'no retina';

fi

#python_file_list=(full_pipeline_active_learning)
#multi_strategy_list=(false)
#suffix_list=(al)
#clear_hist_list=(true)


<<cmd
python_file_list=(full_pipeline_suggest_labels full_pipeline full_pipeline_tars full_pipeline_o2u full_pipeline_active_learning)
multi_strategy_list=(true false false false false)
suffix_list=(sl infl tars o2u al)
clear_hist_list=(true false false false true)
default_strategy_list=(0 0 3 0 0)
cmd

python_file_list=(full_pipeline_suggest_labels)
multi_strategy_list=(true)
suffix_list=(sl${source_file_suffix})
clear_hist_list=(false)
default_strategy_list=(0)


<<cmd
python_file_list=(full_pipeline_suggest_labels full_pipeline_active_learning)
multi_strategy_list=(true false)
suffix_list=(sl ac)
clear_hist_list=(true true)
default_strategy_list=(0 0)
cmd


for (( i = 0 ; i < ${total_iteration} ; i++ ))
do

	echo "start neural model"

	curr_python_file=full_pipeline_suggest_labels

	multi_strategy=false

        curr_suffix=neural_sl

        curr_clear_hist=true

        default_strategy=0


#	echo "bash run_full_pipeline_suggest_label_repeat_all_neural.sh ${hist_period}  ${1} $2 ${remove_count} ${repeat_num} $lr ${l2_decay} ${epochs} ${batch_size} ${curr_clear_hist} ${i} ${isGPU} ${GPUID} ${derived_lr} ${regular_rate} ${curr_python_file} ${multi_strategy} ${curr_suffix} ${default_strategy} ${source_file}_neural ${model} false"

 #               bash run_full_pipeline_suggest_label_repeat_all_neural.sh ${hist_period}  ${1} $2 ${remove_count} ${repeat_num} $lr ${l2_decay} ${epochs} ${batch_size} ${curr_clear_hist} ${i} ${isGPU} ${GPUID} ${derived_lr} ${regular_rate} ${curr_python_file} ${multi_strategy} ${curr_suffix} ${default_strategy} ${source_file}_neural ${model} false


	for k in "${!python_file_list[@]}"; do
		curr_python_file=${python_file_list[$k]}

		multi_strategy=${multi_strategy_list[$k]}

		curr_suffix=${suffix_list[$k]}

		curr_clear_hist=${clear_hist_list[$k]}

		default_strategy=${default_strategy_list[$k]}

		echo "${k} ${curr_python_file} ${multi_strategy}"
#<<cmd
        	echo "bash run_full_pipeline_suggest_label_repeat_all.sh ${hist_period}  ${1} $2 ${remove_count} ${repeat_num} $lr ${l2_decay} ${epochs} ${batch_size} ${curr_clear_hist} ${i} ${isGPU} ${GPUID} ${derived_lr} ${regular_rate} ${curr_python_file} ${multi_strategy} ${curr_suffix} ${default_strategy} ${source_file} ${model} false ${is_tars}"

	        bash run_full_pipeline_suggest_label_repeat_all.sh ${hist_period}  ${1} $2 ${remove_count} ${repeat_num} $lr ${l2_decay} ${epochs} ${batch_size} ${curr_clear_hist} ${i} ${isGPU} ${GPUID} ${derived_lr} ${regular_rate} ${curr_python_file} ${multi_strategy} ${curr_suffix} ${default_strategy} ${source_file} ${model} false ${is_tars}
<<cmd
		if [ "${dataset_name}" == 'retina' ];
	        then

#        	        rm ${dataset_dir}/${dataset_name}/*${curr_suffix}*
#			rm ${dataset_dir}/${dataset_name}/*al*

#                rm ${dataset_dir}/${dataset_name}/Logistic_regression_influence_*

 #               rm ${dataset_dir}/${dataset_name}/full_existing_labeled_id_tensor_sl_v*

	        else

#        	        rm ${dataset_dir}/*${curr_suffix}*
#			rm ${dataset_dir}/*al*

  #              rm ${dataset_dir}/Logistic_regression_influence_*

   #             rm ${dataset_dir}/full_existing_labeled_id_tensor_sl_v*

	        fi
cmd
	done
<<cmd
	cd ../iterative_detect/
	

	total_removed_count=$(( remove_count*repeat_num  ))

	model=Logistic_regression

	echo "python3 full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${total_removed_count} --output_dir ${dataset_dir} --start > ${dataset_dir}/output_start_sl_once.txt 2>&1"


	python3 full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${total_removed_count} --output_dir ${dataset_dir} --start  > ${dataset_dir}/output_start_sl_once.txt 2>&1

	cd ../script/

	echo "bash run_full_pipeline_suggest_label0.sh ${1} $2 ${remove_count} ${repeat_num} $lr ${l2_decay} ${epochs} ${batch_size} false ${i} ${isGPU} ${GPUID}"

        bash run_full_pipeline_suggest_label0.sh ${1} $2 ${remove_count} ${repeat_num} $lr ${l2_decay} ${epochs} ${batch_size} false ${i} ${isGPU} ${GPUID}


	if [ "${dataset_name}" == 'retina' ];
        then

                rm ${dataset_dir}/${dataset_name}/full_training_noisy_dataset_v*

                rm ${dataset_dir}/${dataset_name}/Logistic_regression_influence_*

                rm ${dataset_dir}/${dataset_name}/full_existing_labeled_id_tensor_v*

        else

                rm ${dataset_dir}/full_training_noisy_dataset_v*

                rm ${dataset_dir}/Logistic_regression_influence_*

                rm ${dataset_dir}/full_existing_labeled_id_tensor_v*

        fi


	
	echo "bash run_full_pipeline_${dataset_name}.sh ${1} $2 ${remove_count} ${repeat_num} $lr ${l2_decay} ${epochs} ${batch_size} false ${i} ${isGPU} ${GPUID}"

        bash run_full_pipeline_${dataset_name}.sh ${1} $2 ${remove_count} ${repeat_num} $lr ${l2_decay} ${epochs} ${batch_size} false ${i} ${isGPU} ${GPUID}
cmd
	if [ "${dataset_name}" == 'retina' ];
	then
		cp ${dataset_dir}/${dataset_name}/random_ids_multi_super_iterations ${dataset_dir}/${dataset_name}/random_ids_multi_super_iterations_v${i};

	        cp ${dataset_dir}/${dataset_name}/model_initial ${dataset_dir}/${dataset_name}/model_initial_v${i};

        	cp ${dataset_dir}/${dataset_name}/w_list_initial ${dataset_dir}/${dataset_name}/w_list_initial_v${i}; 


	        cp ${dataset_dir}/${dataset_name}/grad_list_initial ${dataset_dir}/${dataset_name}/grad_list_initial_v${i};

		rm ${dataset_dir}/${dataset_name}/full_training_noisy_dataset_${curr_suffix}_v*

		rm ${dataset_dir}/${dataset_name}/Logistic_regression_influence_*

		rm ${dataset_dir}/${dataset_name}/full_existing_labeled_id_tensor_${curr_suffix}_v*

	else

		cp ${dataset_dir}/random_ids_multi_super_iterations ${dataset_dir}/random_ids_multi_super_iterations_v${i};

		cp ${dataset_dir}/model_initial ${dataset_dir}/model_initial_v${i};

		cp ${dataset_dir}/w_list_initial ${dataset_dir}/w_list_initial_v${i}; 


		cp ${dataset_dir}/grad_list_initial ${dataset_dir}/grad_list_initial_v${i};

		rm ${dataset_dir}/full_training_noisy_dataset_${curr_suffix}_v*

		rm ${dataset_dir}/Logistic_regression_influence_*

		rm ${dataset_dir}/full_existing_labeled_id_tensor_${curr_suffix}_v*

	fi


done

