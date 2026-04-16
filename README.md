Install python 3.11.2, 64 AMD

```
py -3.11-64 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Go to [this folder](https://drive.google.com/drive/u/1/folders/1z02puhXXChU2A3bWjqrZm-SmDbX3m3aE) to download relevant files, including LLM output labels. Speaking of which, I used Llama 3 via Ollama since that's a free one, but you'd need to download Ollama and pull the model.


Despite calling `predict_animal_products`, the original LTS prompt in `labeling.py` actually relates to shark products, which confuses which command line use cases to input. 

```
python main_cluster.py -task "sharks" -training_data_path "data/training_sharks.csv" -validation_data_path "data/validation_sharks.csv" -sample_size 200 -sampling "thompson" -balance True -filter_label True -labeling_llm "ollama" -model_path "bert-base-uncased" -metric "f1" -metric_baseline 0.5 -cluster_size 10 -loop_size 10
```




