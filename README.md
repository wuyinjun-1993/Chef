# CHEF: A Cheap and Fast Pipeline for Iteratively Cleaning Label Uncertainties 
The implementation of the system *CHEF* from the following paper:

> [CHEF: A Cheap and Fast Pipeline for Iteratively Cleaning Label Uncertainties](https://arxiv.org/abs/2107.08588)\
Yinjun Wu, James Weimer, Susan B. Davidson\
47th International Conference on Very Large Data Bases ([VLDB](https://https://vldb.org/2021/)), 2021\
_arXiv:2107.08588_


## Getting Started
This project is implemented with Python3. To start using this project, it is essential to run the following steps

### Installing the prerequisites
To install the required python packages, we can first clone a local copy of this project
```
git clone https://github.com/thuwuyinjun/Chef.git
```
and then run the following commands with PIP inside the project directory:
```
cd Chef/
pip install -r requirements.txt
```

## Usage
To show how to use *CHEF* for the label cleaning tasks, we use the **Twitter sentiment analysis dataset** (**Twitter dataset** for short hereafter) as the running example. The original version of the **Twitter dataset** is available at [here](https://github.com/naimulhuq/Capstone/blob/master/Data/Airline-Full-Non-Ag-DFE-Sentiment%20(raw%20data).csv) which includes the non-aggregated labels provided by different human annotators. Suppose the project directory is '/path/to/dir', then we also provide a copy of the **Twitter dataset** in the directory '/path/to/dir/data/twitter/'.

### Pre-process data:
This step aims at 1) transforming the plain-text data into the embedding representations; 2) automatically derive suitable **labeling functions** and the **probabilistic labels** for the samples without ground-truth labels. To pre-process the **Twitter dataset**, we need to utilize the implementation of the project [Interactive Weak Supervision](https://github.com/benbo/interactive-weak-supervision) for deriving the labeling functions and the probabilistic labels. We copied the code of [Interactive Weak Supervision](https://github.com/benbo/interactive-weak-supervision) in the directory '/path/to/dir/interactive_weak_supervision/'. Then we can run the following commands for pre-processing the **Twitter dataset**:

```
cd /path/to/dir/process_data/
python pre_process_twitter.py --input_data_dir /path/to/dir/data/twitter/ --output_dir /output/dir/
```

The '/output/dir/' is a folder for storing the generated the training set, the validation set and the test set, which are then used for the experiments. The default value of '/output/dir/' is '/path/to/dir/.gitignore/crowdsourced_dataset/twitter/' for the Twitter dataset, which can be changed for any directory name.

### Evaluating the performance of *CHEF*

After pre-processing the datasets that we want, we can then evaluate the performance of *CHEF* on those datasets. Note that *CHEF* is comprised of the following three components: 1) *INFL* for identifying the most **influential** training samples with probabilistic labels, which can then be cleaned by the human annotators; 2) *Increm-INFL* for speeding up the selections of the most influential training samples; 3) *Deltagrad-L* for incrementally updating the models after the labels of the most influential training samples are cleaned. We can then use the following commands to evaluate the performance of those three components respectively.

#### Construct an initial model
In the very beginning, it is essential to construct an initial machine learning model by using all the training samples, including the ones with ground-truth labels and the ones with probabilistic labels. We can achieve this by running the following commands:
```
cd /path/to/dir/iterative_detect/
python3 full_pipeline_infl.py --bz $1 --epochs $2 --tlr $3 --wd $4 --dataset $5 --model $6  --output_dir $7 --start --restart --regular_rate $8 --suffix $9  --no_prov
```
in which, 
$1=mini-batch size for model training (SGD), e.g., 1000,\
$2=number of epochs for model training (SGD), e.g., 400,\
$3=learning rate for model training (SGD), e.g., 0.005,\
$4=L2 regularization rate for model training (SGD), e.g., 0.002,\
$5=dataset name, e.g., twitter,\
$6=model name, e.g., Logistic_regression,\
$7=data directory to store the generated results from the pre-processing step, e.g., '/output/dir/' used above,\
$8=regularization coefficient for regularizing the training samples with probabilistic labels, e.g., 0.8, which is the value of &lambda; in the paper.\
$9=tag for the names of the output files (e.g., output model parameters), which could be any string.

For example, to train a model on the Twitter dataset, the above commands could be instantiated as follows:

```
cd /path/to/dir/iterative_detect/
python3 full_pipeline_infl.py --bz 1000 --epochs 400 --tlr 0.005 --norm loss --wd 0.002 --dataset twitter --model Logistic_regression  --output_dir /home/wuyinjun/pg_data/twitter/ --start --restart --regular_rate 0.5 --suffix sl_initial_zero0  --no_prov
```





#### Evaluating *INFL*:

After the initial model is constructed, we can then use *INFl* to identify the Top-b **influential** training samples for cleaning, in which b is the number of training samples to be cleaned each time. To achieve this, we can then run the following command:


```
cd /path/to/dir/iterative_detect/
python3 full_pipeline_infl.py --derived_lr $1 --derived_epochs $2 --derived_bz $3 --bz $4 --epochs $5 --tlr $6 --wd $7 --dataset $8 --model $9  --removed_count $10 --output_dir $11  --resolve_conflict $12 --regular_rate $13 --suffix $14  --no_prov
```

$1,$2,$3=The learning rate, the number of epochs and the mini-batch size used in the conjugate gradient method for deriving the matrix-vector product <img src="https://render.githubusercontent.com/render/math?math=\nabla_{\textbf{w}} F(\textbf{w},\mathcal{Z}_{\text{val}})^\top \textbf{H}^{-1}(\textbf{w})"> in Equation (6)


$4,$5,$6,$7=mini-batch size, the number of epochs, the learning rates and the L2 regularization rate for SGD iterations\
$8=dataset name, e.g., twitter\
$9=model name, e.g., Logistic_regression\
$10=the number of training samples to be cleaned, i.e., the value of **b**, e.g., 10\
$11=the directory of the generated results from the previous step on constructing the initial model, e.g., '/output/dir/'\
$12=which cleaning strategy if *INFL* is used, 0=strategy one, 3=strategy two, 1=strategy three\
$13=regularization coefficient for regularizing the training samples with probabilistic labels, e.g., 0.8, which is the value of &lambda; in the paper\
$14=tag for the names of the output files (e.g., output model parameters), which could be any string


For example, we can run the following command on the Twitter dataset:

```
cd /path/to/dir/iterative_detect/
python3 full_pipeline_infl.py --derived_lr 0.001 --derived_epochs 50 --derived_bz 2000 --bz 1000 --epochs 400 --tlr 0.005 --wd 0.002 --dataset twitter --model Logistic_regression  --removed_count 20 --output_dir /output/dir/  --resolve_conflict 1 --regular_rate 0.5 --suffix sl_initial_zero0  --no_prov
```

If we want to continuously clean the training dataset, we can add one flag, '--continue_labeling', in the command above, i.e.:
```
python3 full_pipeline_infl.py --derived_lr 0.001 --derived_epochs 50 --derived_bz 2000 --bz 1000 --epochs 400 --tlr 0.005 --wd 0.002 --dataset twitter --model Logistic_regression  --removed_count 20 --output_dir /output/dir/  --resolve_conflict 1 --regular_rate 0.5 --suffix sl_initial_zero0  --no_prov --continue_labeling
```

which can start from the updated models and the updated training dataset next time when this command is executed. 



We also prepared a script to run INFL and compare it against other baseline approaches after the pre-processing step. We can run the following command to achieve this:

```
cd /path/to/dir/script/

bash exp_infl.sh $1 $2 $3 $4 $5 $6 $7
```

$1=output directory used in the pre-processing step, e.g., '/output/dir/'\
$2=dataset name, e.g., twitter \
$3=whether GPU is used, true for yes and false for no \
$4=The GPU ID used for experiments \
$5=model name, e.g., Logistic_regression \
$6=suffix of the file names for storing the hyper-parameters used for the experiments. The full name of the files storing the hyper-parameters will be "hyper_$2$6" \
$7=whether to compare against the baseline 'TARS'


For example, the above command could be instantiated as follows for the Twitter dataset:

```
cd /path/to/dir/script/

bash exp_infl.sh /output/dir/ twitter true 1 Logistic_regression _initial_zero0 false
```



#### Evaluating *Increm-INFL* and *DeltaGrad-L*

To run *Increm-INFL* and *DeltaGrad-L*, we can use the following command:
```
cd /path/to/dir/iterative_detect/
python3 full_pipeline_infl.py --derived_lr $1 --derived_epochs $2 --derived_bz $3 --bz $4 --epochs $5 --tlr $6 --wd $7 --dataset $8 --model $9  --removed_count $10 --output_dir $11  --resolve_conflict $12 --regular_rate $13 --suffix $14  --no_prov --incremental --hist_period $15 --continue_labeling
```

in which the flag '--incremental' is used for indicating the use of the *Increm-INFL* and *DeltaGrad-L*. During the first time when the above command is executed, we need to cache some provenance information, which happens every $15 SGD epochs. For example, $15=20.


We also prepared a script to run Increm-INFL and DeltaGrad-L and compared them against the case where they are not used, for which we can run the following command:
```
cd /path/to/dir/script/
bash exp_compare_incre.sh $1 $2 $3 $4 $5
```

$1=output directory used in the pre-processing step, e.g., '/output/dir/'\
$2=dataset name, e.g., twitter \
$3=whether GPU is used, true for yes and false for no \
$4=The GPU ID used for experiments \
$5=suffix of the file names for storing the hyper-parameters used for the experiments. The full name of the files storing the hyper-parameters will be "hyper_$2$5" \









## citation
Please cite our paper if you use code from this repo:

```
@article{wu2021chef,
  title={CHEF: A Cheap and Fast Pipeline for Iteratively Cleaning Label Uncertainties},
  author={Wu, Yinjun and Weimer, James and Davidson, Susan B.},
  journal={Proceedings of the VLDB Endowment},
  volume={14},
  number={11},
  year={2021},
  publisher={VLDB Endowment}
}
```
