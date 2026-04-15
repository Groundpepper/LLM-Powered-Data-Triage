import os
import pandas as pd
import numpy as np

from typing import Any
from scipy.stats import beta


class ThompsonSampler:
    def __init__(self, n_bandits, alpha=0.5, beta=0.5, decay=0.99):
        self.n_bandits = n_bandits
        self.wins = np.zeros(n_bandits)  # Initialize wins array
        self.losses = np.zeros(n_bandits)  # Initialize losses array
        self.alpha = alpha  # Prior parameter for Beta distribution (successes)
        self.beta = beta   # Prior parameter for Beta distribution (failures)
        self.decay = decay
        
        if os.path.exists('selected_ids.txt'):
            os.remove('selected_ids.txt')
        self.selected_ids = set()
    
        if os.path.exists('wins.txt'):
            os.remove('wins.txt')
        self.wins = np.zeros(n_bandits)        

        if os.path.exists('losses.txt'):
            os.remove('losses.txt')
        self.losses = np.zeros(n_bandits)

    def choose_bandit(self):
        betas = beta(self.wins + self.alpha, self.losses + self.beta)
        sampled_rewards = betas.rvs(size=self.n_bandits)
        return np.argmax(sampled_rewards)

    def update(self, chosen_bandit, reward_difference):
        if reward_difference > 0: self.wins[chosen_bandit] += 1
        else: self.losses[chosen_bandit] += 1

        self.wins *= self.decay
        self.losses *= self.decay
        np.savetxt('wins.txt', self.wins)
        np.savetxt('losses.txt', self.losses)

    def select_data(self, df, chosen_bandit, sample_size):
        filtered_df = df[df['label_cluster'] == chosen_bandit].sample(min(sample_size, len(df[df['label_cluster'] == chosen_bandit])))
        return filtered_df
    
    def get_sample_data(self, df, sample_size, filter_label: bool, trainer: Any):
        if 'id' not in df.columns:
            df = df.reset_index().rename(columns={'index': 'id'})
        df = df[~df['id'].isin(self.selected_ids)]

        data = pd.DataFrame()
        while data.empty:
            chosen_bandit = self.choose_bandit()
            bandit_df = df[df["label_cluster"] == chosen_bandit]
            print(f"Chosen bandit {chosen_bandit}, length {len(bandit_df)}")
            if not bandit_df.empty:
                if filter_label:
                    if trainer.get_clf():
                        bandit_df["predicted_label"] = trainer.get_inference(bandit_df)
                    if "predicted_label" in bandit_df.columns:
                        print(f"inference results: {bandit_df['predicted_label'].value_counts()}")
                        pos = bandit_df[bandit_df["predicted_label"] == 1]
                        neg = bandit_df[bandit_df["predicted_label"] == 0]
                        if pos.empty:
                            print("no positive data available")
                            data = pos
                        else:
                            n_sample = sample_size / 2
                            data = self.select_data(pos, chosen_bandit, int(n_sample))
                            neg_data = self.select_data(neg, chosen_bandit, int(sample_size - len(data)))
                            data = pd.concat([data, neg_data]).sample(frac=1)
                    else:
                        data = self.select_data(bandit_df, chosen_bandit, sample_size)
                else:
                    data = self.select_data(bandit_df, chosen_bandit, sample_size)

        self.selected_ids.update(data['id'])
        with open('selected_ids.txt', 'w') as f:
            f.write('\n'.join(map(str, self.selected_ids)))        
        return data, chosen_bandit
