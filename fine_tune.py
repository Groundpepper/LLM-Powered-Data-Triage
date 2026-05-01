import torch
import pandas as pd
from typing import Any, Optional, Dict, Union, Tuple, Callable, List
from datasets import Dataset, DatasetDict
from transformers import BertTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, BertForSequenceClassification, PreTrainedModel, TrainerCallback
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score, recall_score, f1_score

import warnings
from sklearn.exceptions import UndefinedMetricWarning
from transformers.utils import logging
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
hf_logging.set_verbosity(50)
logging.set_verbosity_error()
logging.set_verbosity_warning()
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class BertFineTuner:
    def __init__(self, model_name: Optional[str], training_data: Optional[pd.DataFrame], test_data: Optional[pd.DataFrame],
            learning_rate=2e-5, dropout=0.2, num_labels=2):
        self.base_model = model_name
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.last_model_metric_results: Dict[str, str] = None
        self.training_data = training_data
        self.test_data = test_data
        self.trainer = None
        self.run_clf = False
        self.learning_rate = learning_rate
        self.weight_decay = 0.0
        self.num_labels = num_labels
        if dropout:
            model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            model.config.hidden_dropout_prob = dropout
            self.model = model
        else:
            self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(self.device)

    def set_clf(self, set: bool):
        self.run_clf = set

    def set_train_data(self, train):
        self.training_data = train

    def tokenize_function(self, element):
        return self.tokenizer(element['title'], padding="max_length", truncation=True, max_length=256)
    
    def create_dataset(self, train, test):
        dataset_train = Dataset.from_pandas(train[["title", "label"]])
        dataset_val = Dataset.from_pandas(test[["title", "label"]])

        dataset = DatasetDict()
        dataset["train"] = dataset_train
        dataset["val"] = dataset_val
        tokenized_data = dataset.map(self.tokenize_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        return tokenized_data, data_collator

    def create_test_dataset(self, df: pd.DataFrame) -> Dataset:
        test_dataset = Dataset.from_pandas(df[["title"]])
        dataset = DatasetDict()
        dataset["test"] = test_dataset
        tokenized_data = dataset.map(self.tokenize_function, batched=True)
        return tokenized_data

    @staticmethod
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average="macro")
        recall = recall_score(labels, preds, average="macro")
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
        metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, 'f1': f1.mean()}
        for i in range(len(f1)): metrics[f"f1_class_{i}"] = f1[i]
        return metrics

    def train_data(self, df, still_unbalanced=False):
        torch.cuda.empty_cache()
        early_stopping_callback = EarlyStoppingCallback(patience=5)
        tokenized_data, data_collator = self.create_dataset(df, self.test_data)

        training_args = TrainingArguments(output_dir="results", eval_strategy="epoch", save_strategy="epoch",
            metric_for_best_model="eval_accuracy", per_device_train_batch_size=32, per_device_eval_batch_size=32,
            num_train_epochs=20, learning_rate=self.learning_rate, weight_decay=self.weight_decay, save_total_limit=2,
            logging_strategy='no', push_to_hub=False, load_best_model_at_end=True, disable_tqdm=True, report_to="none")

        trainer = Trainer(model=self.model, args=training_args, data_collator=data_collator,
            compute_metrics=BertFineTuner.compute_metrics, train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["val"], callbacks=[early_stopping_callback, CleanEvalPrintCallback()])
        
        trainer.train()
        results = trainer.evaluate()
        self.trainer = trainer
        return results, self.trainer


    def get_inference(self, df: pd.DataFrame) -> torch.Tensor:
        predicted_labels = []
        chunk_size = 10000
        total_records = len(df)
        start_index = 0

        while start_index < total_records:
            end_index = min(start_index + chunk_size, total_records)
            chunk = df[start_index:end_index]
            test_dataset = self.create_test_dataset(chunk)
            predictions = self.trainer.predict(test_dataset["test"])
            prediction_scores = predictions.predictions
            batch_predicted_labels = torch.argmax(torch.tensor(prediction_scores), dim=1)
            predicted_labels.append(batch_predicted_labels)
            start_index = end_index

        predicted_labels = torch.cat(predicted_labels)
        return predicted_labels

    def save_model(self, path: str):
        self.trainer.save_model(path)

    def update_model(self, model_name, model_metric_results):
        self.last_model_metric_results = {model_name: model_metric_results}
        self.base_model = model_name


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=5, log_dir=None):
        self.patience = patience
        self.best_loss = float('inf')
        self.wait = 0
        self.log_dir = log_dir

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if state.is_world_process_zero and state.log_history:
            current_loss = None
            for log_entry in reversed(state.log_history):
                if 'eval_loss' in log_entry:
                    current_loss = log_entry['eval_loss']
                    break
            if current_loss is not None:
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        control.should_training_stop = True
                if self.log_dir:
                    with open(f"{self.log_dir}/epoch_{state.epoch}.txt", "w") as f:
                        for log in state.log_history:
                            f.write(f"{log}\n")


class CleanEvalPrintCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None: return

        print('\n')
        print(f"Accuracy: {metrics['eval_accuracy']:.4f}")
        print(f"Precision: {metrics['eval_precision']:.4f}")
        print(f"Recall: {metrics['eval_recall']:.4f}")
        print(f"F1: {metrics['eval_f1']:.4f}")        
        i = 0
        while f"eval_f1_class_{i}" in metrics:
            print(f"F1 class {i}: {metrics[f'eval_f1_class_{i}']:.4f}")
            i += 1
        print('\n')
