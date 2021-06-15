# Chef

Pre-process data (suppose the project director is '/path/to/dir'):

cd process_data

python pre_process_twitter --output_dir /path/to/dir/data/twitter/


To compare infl against other baseline methods, run the following command on twitter dataset:

cd script/

bash exp2_full.sh /path/to/dir/data/twitter/ twitter true 1 Logistic_regression '' false
