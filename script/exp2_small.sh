#!/bin/bash



python_cmd=python3


#cd ../iterative_detect/


#batch_size=$8

#epochs=$7

#lr=$5


#l2_decay=$6


#outer_loop_count=2000


dataset_name=$2

model=Logistic_regression


remove_count=500

dataset_dir=$1




repeat_num=1

duti_training_lr=200

#clear_hist=$9

total_iteration=1

#running_iteration=${10}

isGPU=${3}

GPUID=${4}


echo "run DUTI"

source ./hyper_${dataset_name}

echo "lr::"$lr
echo "l2 decay::" ${l2_decay}
echo "epochs::" ${epochs}
echo "batch size::" ${batch_size}


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



for (( i = 0 ; i < ${total_iteration} ; i++ ))
do

#	cp ${dataset_dir}/random_ids_multi_super_iterations_v${i} ${dataset_dir}/random_ids_multi_super_iterations

#<<cmd
        echo "bash run_full_pipeline_suggest_label.sh ${hist_period}  ${1} $2 ${remove_count} ${repeat_num} $lr ${l2_decay} ${epochs} ${batch_size} true ${i} ${isGPU} ${GPUID}"

        bash run_full_pipeline_suggest_label_repeat_sub_small.sh ${hist_period}  ${1} $2 ${remove_count} ${repeat_num} $lr ${l2_decay} ${epochs} ${batch_size} true ${i} ${isGPU} ${GPUID}
<<cmd
	cd ../iterative_detect/
	

	total_removed_count=$(( remove_count*repeat_num  ))

	model=Logistic_regression

	echo "python3 full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${total_removed_count} --output_dir ${dataset_dir} --start > ${dataset_dir}/output_start_sl_once.txt 2>&1"


	python3 full_pipeline_suggest_labels.py --lr 2 --bz ${batch_size} --epochs ${epochs} --tlr ${lr} --norm loss --wd ${l2_decay} --dataset ${dataset_name} --model ${model}  --removed_count ${total_removed_count} --output_dir ${dataset_dir} --start  > ${dataset_dir}/output_start_sl_once.txt 2>&1

	cd ../script/
cmd
	echo "bash run_full_pipeline_suggest_label0.sh ${1} $2 ${remove_count} ${repeat_num} $lr ${l2_decay} ${epochs} ${batch_size} false ${i} ${isGPU} ${GPUID}"

        bash run_full_pipeline_suggest_label0.sh ${1} $2 ${remove_count} ${repeat_num} $lr ${l2_decay} ${epochs} ${batch_size} false ${i} ${isGPU} ${GPUID}
	
	echo "bash run_full_pipeline.sh ${1} $2 ${remove_count} ${repeat_num} $lr ${l2_decay} ${epochs} ${batch_size} false ${i} ${isGPU} ${GPUID}"

        bash run_full_pipeline.sh ${1} $2 ${remove_count} ${repeat_num} $lr ${l2_decay} ${epochs} ${batch_size} false ${i} ${isGPU} ${GPUID}

	cp ${dataset_dir}/random_ids_multi_super_iterations ${dataset_dir}/random_ids_multi_super_iterations_v${i}


done

