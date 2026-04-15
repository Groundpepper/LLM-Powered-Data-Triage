import argparse
import os
import nltk
import json
import time
import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from labeling import Labeling
from random_sampling import RandomSampler
from preprocessing import TextPreprocessor
from fine_tune import BertFineTuner
from thompson_sampling import ThompsonSampler
from LDA import LDATopicModel

nltk.download('punkt')


def main():
    """ arguments handling """    
    parser = argparse.ArgumentParser(prog="Sampling fine-tuning", description='Perform Sampling and fine tune')
    parser.add_argument('-task', type=str, required=True, help="Name of task")
    parser.add_argument('-training_data_path', type=str, required=True, help="The initial file to be used")
    parser.add_argument('-validation_data_path', type=str, required=True, help="path to validation")
    parser.add_argument('-sample_size', type=int, required=True, help="sample size")
    parser.add_argument('-sampling', type=str, required=True, help="Name of sampling method")
    parser.add_argument('-filter_label', type=bool, required=True, help="use model clf results to filter data")
    parser.add_argument('-balance', type=bool, required=True, help="balance positive and neg sample")
    parser.add_argument('-labeling_llm', type=str, required=True, help="Model to be used for labeling or file if label already on file")
    parser.add_argument('-model_path', type=str, required=True, help="model base for fine tune")
    parser.add_argument('-metric', type=str, required=True, help="The type of metric to be used for baseline")
    parser.add_argument('-metric_baseline', type=float, required=True, help="The initial baseline metric")
    parser.add_argument('-cluster_size', type=str, required=True, help="path to validation")
    parser.add_argument('-loop_size', type=str, required=True, help="number of iterations in training loop")
    
    args = parser.parse_args()
    task = args.task
    training_data_path = args.training_data_path
    validation_data_path = args.validation_data_path
    sample_size = args.sample_size
    sampling = args.sampling
    balance = args.balance
    filter_label = args.filter_label
    labeling_llm = args.labeling_llm
    model_path = args.model_path
    metric = args.metric
    metric_baseline = args.metric_baseline
    cluster_size = args.cluster_size
    loop_size = args.loop_size
    
    """ folders handling """
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("log", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    """ dealing with validation and training data csvs """
    preprocessor = TextPreprocessor()
    validation = pd.read_csv(validation_data_path)
    validation["training_text"] = validation["title"]
    data = None
    try:
        # if cluster is already labeled, do this
        data = pd.read_csv(training_data_path.split('.')[0] + "_lda.csv")
        n_cluster = data['label_cluster'].value_counts().count()
        print(f"Using data saved on disk: {n_cluster} clusters")
    except Exception:
        # if cluster is not labled yet, do this
        print("Creating LDA")
        data = pd.read_csv(training_data_path)
        data = preprocessor.preprocess_df(data)
        lda_topic_model = LDATopicModel(num_topics=cluster_size)
        topics = lda_topic_model.fit_transform(data['title'].to_list())
        data["label_cluster"] = topics
        n_cluster = data['label_cluster'].value_counts().count()
        data.to_csv(training_data_path.split('.')[0] + "_lda.csv", index=False)
        print(f"LDA created: {n_cluster} clusters")

    """ select LTS-text model (the only model available atp) """
    trainer = BertFineTuner(model_path, None, validation)
    if trainer.device == 'cuda:0': print('Using cuda!')
    else: print('Using CPU')
    
    """ get labelling object with LLM prompting functionalities """
    labeler = Labeling(label_model=labeling_llm)
    labeler.set_model()

    """ get sampler """
    sampler = None
    if sampling == "thompson": sampler = ThompsonSampler(n_cluster)
    elif sampling == "random": sampler = RandomSampler(n_cluster)
    else: raise ValueError("Choose one of thompson or random")

    """ train loop_size times """
    df = None
    start = time.time()
    if not loop_size: loop_size = 10
    for i in range(int(loop_size)):
        """ sample data; DELETE labeled.csv if model use change """
        labeled_data_path = f"{training_data_path.split('.')[0]}_labeled.csv"
        labeled_data = None
        chosen_bandit = sampler.choose_bandit()

        print(f'Trying to retrieve already sampled data for bandit {chosen_bandit}...')
        if not os.path.exists(labeled_data_path):
            pd.DataFrame(columns=['id', 'title', 'training_text', 'text', 'answer', 'label', 'chosen_bandit']).to_csv(labeled_data_path, index=False)
            
        labeled_data = pd.read_csv(labeled_data_path)
        labeled_data = labeled_data.drop_duplicates(subset=['id'], keep='last')
        labeled_data = labeled_data[labeled_data['answer'].isin(['not a relevant animal', 'relevant animal'])]
        labeled_data.to_csv(labeled_data_path, index=False)
        labeled_data = labeled_data[labeled_data['chosen_bandit'] == chosen_bandit]

        # if previously sampled in the same run
        if os.path.exists('selected_ids.txt'):
            seen_data_list = np.loadtxt('selected_ids.txt', delimiter=None, dtype=str, skiprows=0)
            labeled_data = labeled_data[~labeled_data['id'].isin(seen_data_list)]

        """ LLM labeling """
        if len(labeled_data) >= sample_size:
            print(f'Sampled data found for bandit {chosen_bandit}')
            df = labeled_data.sample(n=sample_size)
            df = df[['id', 'title', 'training_text', 'text', 'answer', 'label']]
            with open('selected_ids.txt', 'w') as f:
                f.write('\n'.join(map(str, df.id)))
        else:
            print(f'No data found for bandit {chosen_bandit}. Rechoosing bandit...')
            sample_data, chosen_bandit = sampler.get_sample_data(data, sample_size, filter_label, trainer)
            df = labeler.generate_inference_data(sample_data, 'title', task)
            df["answer"] = df.apply(lambda x: labeler.predict_outcome(x, task), axis=1)
            df["answer"] = df["answer"].str.strip()
            df["label"] = np.where(df["answer"] == 'relevant animal', 1, 0)
            df['chosen_bandit'] = chosen_bandit
            print(f'inference df: {df["label"].value_counts()}')

            if os.path.exists(labeled_data_path):
                labeled_data = pd.read_csv(labeled_data_path)
                labeled_data = pd.concat([labeled_data, df])
                labeled_data.to_csv(labeled_data_path, index=False)
            else:
                df.to_csv(labeled_data_path, index=False)
            df = df[['id', 'title', 'training_text', 'text', 'answer', 'label']]
        
        """ balance sample if needed """
        if balance:
            if len(df[df["label"] == 1]) > 0:
                unbalanced = len(df[df["label"] == 0]) / len(df[df["label"] == 1]) > 2
                if unbalanced:
                    label_counts = df["label"].value_counts()
                    min_count = min(label_counts)
                    balanced_df = pd.concat([df[df["label"] == 0].sample(min_count * 2), df[df["label"] == 1].sample(min_count)])
                    df = balanced_df.sample(frac=1).reset_index(drop=True)
                    print(f"Balanced data: {df.label.value_counts()}")
            else:
                raise Exception('positive class (1) not found, try with larger sample sizes')

        """ checks """
        model_name = trainer.get_base_model()
        model_results = trainer.get_last_model_acc()
        if model_results: metric_baseline = model_results[model_name]
        print(f"Model {metric} metric is currently {metric_baseline}")

        """ recheck balance status """
        still_unbalanced = len(df[df["label"] == 0]) / len(df[df["label"] == 1]) >= 2
        if still_unbalanced: print(f"Unbalanced? {still_unbalanced}")

        """ training """
        results, huggingface_trainer = trainer.train_data(df, still_unbalanced)
        reward_difference = results[f"eval_{metric}"] - metric_baseline
        print(f"Model changed by {reward_difference}. Save model? {reward_difference > 0}")
        if reward_difference > 0:
            model_name = f"models/finetuned_{i}_bandit_{chosen_bandit}"
            trainer.update_model(model_name, results[f"eval_{metric}"], save_model=True)
            additional_training_data_path = f"{training_data_path.split('.')[0]}_training_data.csv"
            if os.path.exists(additional_training_data_path):
                train_data = pd.read_csv(additional_training_data_path)
                df = pd.concat([train_data, df])
            df.to_csv(additional_training_data_path, index=False)
            if os.path.exists('positive_data.csv'): os.remove('positive_data.csv')
            if filter_label: trainer.set_clf(True)
        else:
            trainer.update_model(model_name, metric_baseline, save_model=False)
            if os.path.exists('positive_data.csv'):
                positive = pd.read_csv("positive_data.csv")
                df = df[df["label"] == 1]
                df = pd.concat([df, positive])
                df = df.drop_duplicates()
            df[df["label"] == 1].to_csv("positive_data.csv", index=False)

        """ save results into record """
        model_results_path = f"{training_data_path.split('.')[0]}_model_results.json"
        if os.path.exists(model_results_path):
            with open(model_results_path, 'r') as file:
                existing_results = json.load(file)
        else: existing_results = {}
        if existing_results.get(str(chosen_bandit)): existing_results[str(chosen_bandit)].append(results)
        else: existing_results[str(chosen_bandit)] = [results]
        with open(model_results_path, 'w') as file:
            json.dump(existing_results, file, indent=4)

        """ update sampler """
        if sampling == "thompson": sampler.update(chosen_bandit, reward_difference)

    print(f'Finished in {(time.time() - start) / 60} minutes')

if __name__ == "__main__":
    main()
