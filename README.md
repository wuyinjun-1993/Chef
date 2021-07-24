# The implementation of the paper "CHEF: A Cheap and Fast Pipeline for Iteratively Cleaning Label Uncertainties" accepted by VLDB 2021. The technical report is available [here](https://arxiv.org/abs/2107.08588)


## Getting Started
This project is implemented with Python3. To start using this project, it is essential to run the following steps

### Installing the prerequisites
To install the required python packages, we can first clone a local copy of this project
```
git clone https://github.com/thuwuyinjun/Chef.git
```



and then run the following commands with PIP:
```
pip install -r requirements.txt
```



Pre-process data (suppose the project director is '/path/to/dir'):

cd process_data

python pre_process_twitter --output_dir /path/to/dir/data/twitter/


To compare infl against other baseline methods, run the following command on twitter dataset:

cd script/

bash exp2_full.sh /path/to/dir/data/twitter/ twitter true 1 Logistic_regression '' false
