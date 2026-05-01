import numpy as np
import pandas as pd
from typing import Any
from tabulate import tabulate


class DirichletMultinomialSampler:
    def __init__(self, n_bandits, n_classes, prior=0.5, decay=0.95):
        self.n_bandits = n_bandits
        self.n_classes = n_classes
        self.prior = prior
        self.decay = decay
        self.alpha = np.full((n_bandits, n_classes), prior)
        self.visit_counts = np.zeros(n_bandits)
        self.selected_ids = set()

    def choose_bandit(self, class_weights=None):
        if class_weights is None: class_weights = np.ones(self.n_classes) / self.n_classes
        samples = np.array([np.random.dirichlet(self.alpha[k]) for k in range(self.n_bandits)])
        scores = samples @ class_weights
        
        print("\nalpha concentrations per bandit:")
        print(tabulate(self.alpha, headers=range(self.n_classes)))
        print("visit counts:", self.visit_counts)
        print("scores:", np.round(scores, 4))
        return int(np.argmax(scores))

    def update(self, chosen_bandit, new_f1_per_class, old_f1_per_class):
        delta = new_f1_per_class - old_f1_per_class
        for c in range(self.n_classes):
            if delta[c] > 0: self.alpha[chosen_bandit][c] += delta[c]
            else: self.alpha[chosen_bandit][c] = max(self.alpha[chosen_bandit][c] + delta[c], self.prior)
    
        visited = self.visit_counts > 0
        self.alpha[visited] = np.maximum(self.prior + (self.alpha[visited] - self.prior) * self.decay, self.prior)
        self.visit_counts[chosen_bandit] += 1 

    def select_data(self, df, chosen_bandit, sample_size):
        filtered_df = df[df['label_cluster'] == chosen_bandit]
        return filtered_df.sample(min(sample_size, len(filtered_df)))

    def get_sample_data(self, df, sample_size, trainer: Any, current_f1_per_class=None):
        if 'id' not in df.columns: df = df.reset_index().rename(columns={'index': 'id'})
        df = df[~df['id'].isin(self.selected_ids)]

        class_weights = None
        if current_f1_per_class is not None:
            weakness = 1 - current_f1_per_class
            class_weights = weakness / weakness.sum()

        data = pd.DataFrame()
        while data.empty:
            chosen_bandit = self.choose_bandit(class_weights=class_weights)
            bandit_df = df[df["label_cluster"] == chosen_bandit]
            print(f"chosen bandit {chosen_bandit}, length {len(bandit_df)}")

            if not bandit_df.empty:
                bandit_df = bandit_df.copy()

                if trainer.run_clf:
                    bandit_df["predicted_label"] = trainer.get_inference(bandit_df)

                if "predicted_label" in bandit_df.columns:
                    classes = bandit_df["predicted_label"].unique()
                    n_per_class = max(1, sample_size // len(classes))
                    samples = []
                    used = set()
                    for cls in classes:
                        cls_df = bandit_df[bandit_df["predicted_label"] == cls]
                        sampled = cls_df.sample(min(n_per_class, len(cls_df)))
                        samples.append(sampled)
                        used.update(sampled.index)
                    data = pd.concat(samples) if samples else pd.DataFrame()
                    missing = sample_size - len(data)
                    if missing > 0:
                        remaining = bandit_df.drop(index=list(used))
                        if len(remaining) > 0:
                            fill = remaining.sample(min(missing, len(remaining)))
                            data = pd.concat([data, fill])
                    data = data.sample(frac=1).reset_index(drop=True)
                else:
                    data = self.select_data(bandit_df, chosen_bandit, sample_size)

        print(data.label.value_counts().sort_index())
        print(f"length {len(data)}")
        self.selected_ids.update(data['id'].tolist())
        return data, chosen_bandit
        