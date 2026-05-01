import argparse
import os
import nltk
import time
import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from labeling import Labeling
from preprocessing import TextPreprocessor
from fine_tune import BertFineTuner
from LDA import LDATopicModel
from dirichlet_sampler import DirichletMultinomialSampler

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

nltk.download('punkt')


def main():
    """ arguments handling """    
    parser = argparse.ArgumentParser(prog="Sampling fine-tuning", description='Perform Sampling and fine tune')
    parser.add_argument('-task', type=str, required=True, help="Name of task")
    parser.add_argument('-training_data_path', type=str, required=True, help="The initial file to be used")
    parser.add_argument('-validation_data_path', type=str, required=True, help="path to validation")
    parser.add_argument('-sample_size', type=int, required=True, help="sample size")
    parser.add_argument('-balance', type=str, required=True, help="balance positive and neg sample")
    parser.add_argument('-labeling_llm', type=str, required=True, help="Model to be used for labeling or file if label already on file")
    parser.add_argument('-model_path', type=str, required=True, help="model base for fine tune")
    parser.add_argument('-metric', type=str, required=True, help="The type of metric to be used for baseline")
    parser.add_argument('-metric_baseline', type=float, required=False, help="The initial baseline metric")
    parser.add_argument('-cluster_type', type=str, required=True, help='LDA or KMeans')
    parser.add_argument('-cluster_size', type=int, required=True, help="path to validation")
    parser.add_argument('-loop_size', type=int, required=True, help="number of iterations in training loop")
    
    args = parser.parse_args()
    task = args.task
    training_data_path = args.training_data_path
    validation_data_path = args.validation_data_path
    sample_size = args.sample_size
    balance = True if args.balance == 'True' else False
    labeling_llm = args.labeling_llm
    model_path = args.model_path
    metric = args.metric
    metric_baseline = args.metric_baseline
    cluster_type = args.cluster_type
    cluster_size = args.cluster_size
    loop_size = args.loop_size
    
    """ folders handling """
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    """ dealing with validation and training data csvs """
    preprocessor = TextPreprocessor()
    validation = pd.read_csv(validation_data_path)
    validation["training_text"] = validation["title"]
    data, data_clustered_path = None, None
    if cluster_type == 'LDA': data_clustered_path = training_data_path.split('.')[0] + f"_lda_{cluster_size}.csv"
    elif cluster_type == 'KMeans': data_clustered_path = training_data_path.split('.')[0] + f"_kmeans_{cluster_size}.csv"
    else: raise Exception('Not implemented.')
    if os.path.exists(data_clustered_path):
        data = pd.read_csv(data_clustered_path)
        n_cluster = data['label_cluster'].value_counts().count()
        print(f"Using data saved on disk: {n_cluster} clusters")
    else:
        print(f"Creating {cluster_type} with {cluster_size} clusters")
        data = pd.read_csv(training_data_path)
        if cluster_type == 'LDA':
            data = preprocessor.preprocess_df(data)
            lda_topic_model = LDATopicModel(num_topics=cluster_size)
            topics = lda_topic_model.fit_transform(data['title'].to_list())
            data["label_cluster"] = topics
        elif cluster_type == 'KMeans':
            model = SentenceTransformer("BAAI/bge-large-en-v1.5")
            texts = data['title'].astype(str).tolist()
            embeddings = model.encode(texts, show_progress_bar=True)
            embeddings = normalize(embeddings)
            kmeans = KMeans(n_clusters=cluster_size)
            data['label_cluster'] = kmeans.fit_predict(embeddings)
            data['clean_title'] = data['title']
        n_cluster = data['label_cluster'].value_counts().count()
        data.to_csv(data_clustered_path, index=False)
        print(f"Clusters created: {n_cluster} clusters")

    """ select LTS-text model (the only model available atp) """
    num_labels = len(validation.label.unique())
    trainer = BertFineTuner(model_path, None, validation, num_labels=num_labels)
    if trainer.device == 'cuda:0': print('Using cuda!')
    else: print('Using CPU')
    
    """ get labelling object with LLM prompting functionalities """
    labeler = Labeling(label_model=labeling_llm, task=task)

    """ get sampler """
    sampler = DirichletMultinomialSampler(n_cluster, num_labels)

    """ train loop_size times """
    old_f1_per_class = np.array([metric_baseline or (1 / num_labels)] * num_labels)
    f1_metrics_per_class, model_name = None, None
    start = time.time()
    for i in range(loop_size):
        """ sample data """
        sample_data, chosen_bandit = sampler.get_sample_data(data, sample_size, trainer, current_f1_per_class=f1_metrics_per_class)
        df = labeler.generate_inference_data(sample_data, 'clean_title')
        df["llm_response"] = labeler.get_llm_responses_parallel(df, verbose=True)
        df['label'] = df['llm_response'].map(labeler.task_pairings[task])
        print(f'inference df: {df["label"].value_counts().sort_index()}')
        print(f'{labeler.failures} / {labeler.attempts} of labels attempts failed')
        
        """ balance sample if needed """
        if balance:
            label_counts = df["label"].value_counts()
            min_count = label_counts.min()
            max_per_class = min_count * 2
            df = df.groupby("label").apply(lambda x: x.sample(min(len(x), max_per_class))).reset_index(drop=True).sample(frac=1)
            print(f"Balanced data: {df.label.value_counts().sort_index()}")

        """ training """
        print(f"Model {metric} metric is currently {old_f1_per_class.mean():.4f} at iteration {i}: {old_f1_per_class}")
        results, x_trainer = trainer.train_data(df)
        f1_metrics_per_class = np.array([results[f"eval_f1_class_{i}"] for i in range(num_labels)])
        reward_per_class = f1_metrics_per_class - old_f1_per_class
        average_reward = reward_per_class.mean()
        print(f"Model changed by {average_reward:.4f}: now {f1_metrics_per_class.mean():.4f}. Save model? {average_reward > 0}")

        """ update sampler with old and new F1 """
        sampler.update(chosen_bandit, f1_metrics_per_class, old_f1_per_class)
        model_name = trainer.base_model
        
        """ save or discard model """
        if average_reward > 0:
            model_name = f"models/loop_{i}_bandit_{chosen_bandit}_{f1_metrics_per_class.mean():.4f}"
            trainer.update_model(model_name, f1_metrics_per_class)
            trainer.set_clf(True)
            old_f1_per_class = f1_metrics_per_class
        else:
            trainer.update_model(model_name, old_f1_per_class)

    trainer.save_model(model_name)
    print(f'Finished in {(time.time() - start) / 3600} hours')

if __name__ == "__main__":
    main()
