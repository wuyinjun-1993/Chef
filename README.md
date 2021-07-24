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

```
cd /path/to/dir/process_data/
python pre_process_twitter --output_dir /path/to/dir/data/twitter/
```

This utilizes the solutions from 


Note that *CHEF* is comprised of three components, i.e. *INFL* for identifying which training samples with probabilistic
### Compare 
To compare infl against other baseline methods, run the following command on twitter dataset:

cd script/

bash exp2_full.sh /path/to/dir/data/twitter/ twitter true 1 Logistic_regression '' false

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
