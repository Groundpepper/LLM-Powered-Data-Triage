Install python 3.11.2, 64 AMD

py -3.11-64 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

The only prompt they gave us for example is the animal products one, so we will have to run that one.

python main_cluster.py -sample_size 200 -filename "data_use_cases/animals" -val_path "data_use_cases/validation_animals.csv"
    -balance True -sampling "thompson" -filter_label True -model_finetune "bert-base-uncased" -labeling "huggingface" -model "text"
    -baseline 0.5 -metric "f1" -cluster_size 10 -loop_size 10



