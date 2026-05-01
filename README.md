Install python 3.11.2, 64 AMD

```
py -3.11-64 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ollama is used for LLM pseudo-labeling. Download it.

```
irm https://ollama.com/install.ps1 | iex
```

Once done, pull relevant models. The best (fastest and most accurate for our purpose) is `glm4:9b`. Also in our report was `llama3:8b` and `granite4.1:8b`.
```
ollama pull glm4:9b
```

Once done, run the model:
```
python main_cluster.py
    -task "emotions"
    -training_data_path "data/training_emotions.csv"
    -validation_data_path "data/validation_emotions.csv"
    -sample_size 400
    -balance True
    -labeling_llm "glm4:9b"
    -model_path "bert-base-uncased"
    -metric "f1"
    -cluster_type KMeans
    -cluster_size 5
    -loop_size 10
```

Do NOT contact For questions or feedback.
