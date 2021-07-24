# The implementation of the paper "CHEF: A Cheap and Fast Pipeline for Iteratively Cleaning Label Uncertainties" accepted by VLDB 2021. The technical report is available [here](https://arxiv.org/abs/2107.08588)


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
To show how to use Chef for the label cleaning tasks, we use the Twitter sentiment analysis dataset (twitter dataset for short hereafter) as the running example. The original version of the twitter dataset is available at [here](https://github.com/naimulhuq/Capstone/blob/master/Data/Airline-Full-Non-Ag-DFE-Sentiment%20(raw%20data).csv) which includes the non-aggregated labels provided by different human annotators. Suppose the project directory is '/path/to/dir', then we also provide a copy of the twitter dataset in the directory '/path/to/dir/data/twitter/'.

### Pre-process data:

```
cd /path/to/dir/process_data/
python pre_process_twitter --output_dir /path/to/dir/data/twitter/
```

To compare infl against other baseline methods, run the following command on twitter dataset:

cd script/

bash exp2_full.sh /path/to/dir/data/twitter/ twitter true 1 Logistic_regression '' false
