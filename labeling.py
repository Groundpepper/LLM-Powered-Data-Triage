import torch
import pandas as pd
import requests
import random
import time

from dask import delayed, compute
from dask.diagnostics import ProgressBar


class Labeling:
    def __init__(self, label_model, task):
        self.label_model = label_model
        self.task = task
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.task_instructions = {
            'sharks': "You are a labeling tool. You must only respond with 'relevant animal' or \
                    'not a relevant animal'. Never ask questions or explain yourself.",
            'emotions': "Answer with one of 'anger', 'fear', 'joy', 'love', 'sadness', or 'surprise'.\
                    Never ask questions or explain yourself.",
        }
        self.task_pairings = {
            'sharks': {'not a relevant animal': 0, 'relevant animal': 1},
            'emotions': {'anger': 0, 'fear': 1, 'joy': 2, 'love': 3, 'sadness': 4, 'surprise': 5}
        }
        self.attempts = 0
        self.failures = 0

    def get_prompt_sharks(self, title):
        return f"""
            You are a labeling tool to create labels for a classification task.
            I will provide text data from an advertisement of a product. The product should be classified as one of two labels:
            
            Label 1: relevant animal - if the product is from any of those 3 animals: Shark, Ray or Chimaeras.
            It should be from a real animal. Not an image or plastic for example.
            
            Label 2: not a relevant animal - if the product is from any other animal that is not Shark, Ray, or Chimaeras,
            or if the product is 100% synthetic (vegan). For products such as teeth, if it's mentioned that is only one tooth,
            you can label it as not a relevant animal. I am only interested in more than one tooth. Also if mentions it's a fossil,
            we are not interested. you can label it as not a relevant animal.
            
            Return only one of the two labels: relevant animal or not a relevant animal, no explanation is necessary.
            
            Exemple:
            1. Advertisement: Great White Shark Embroidered Patch Iron on Patch For Clothes
            Label: not a relevant animal

            The product in example 1 is a piece of clothing with an animal embroidered. The product is not MADE by any animal product.

            2. Advertisement: (sj480-50) 6" White Tip Reef SHARK jaw love sharks jaws teeth Triaenodon obesus
            Label: relevant animal
            
            The product in example 2 is selling a shark jaw. 100% animal product in this case.

            3. Advertisement: Wholesale Group - 20 Perfect 5/8" Modern Tiger Shark Teeth
            Label: relevant animal
            
            In example 3 we have a set of 20 teeth. In this case is True.

            4. Advertisement: Mario Buccellati, a Rare and Exceptional Italian Silver Goat For Sale at 1stDibs
            Label: not a relevant animal
            
            This example 4 is also not an animal product. The goat in the ad is made out of silver and it's not the animal we are interested.
            
            5. Advertisement: HUGE SHARK TOOTH FOSSIL 3&1/4" GREAT Serrations Upper Principal
            Label: not a relevant animal
            
            This is a product from a shark, but is not an animal product because it's only one tooth and it's a fossil.

            6. Advertisement: {title}
            Label:
        """

    def get_prompt_emotions(self, title):
        return f"""
            You are a labeling tool to create labels for a classification task.
            I will provide text data in the form of a sentence. The text should be classified as one of six labels:
            
            Label 1: anger
            Label 2: fear
            Label 3: joy
            Label 4: love
            Label 5: sadness
            Label 6: surprise

            Example:
            1. Text: i feel like taking a whack at someone s eye and spitting on it a cranky old lady i try to cheer myself up
            Label: anger

            2. Text: i for one am feeling a bit anxious at how long we are staying but i know we need to do this
            Label: fear

            3. Text: i feel incredibly lucky just to be able to talk to her
            Label: joy

            4. Text: i miss our talks our cuddling our kissing and the feelings that you can only share with your beloved
            Label: love

            5. Text: im feeling quite sad and sorry for myself but ill snap out of it soon
            Label: sadness

            6. Text: i cant seem to get passed feeling stunned
            Label: surprise
            
            7. Text: {title}
            Label: 
        """

    def generate_prompt(self, title):
        if self.task == 'sharks': return self.get_prompt_sharks(title)
        elif self.task == 'emotions': return self.get_prompt_emotions(title)
        else: raise Exception('Unknown task')

    def generate_inference_data(self, data, column):
        examples = []
        for _, data_point in data.iterrows():
            row = data_point.to_dict()
            row["prompt_input"] = data_point["clean_title"]
            row["full_prompt"] = self.generate_prompt(data_point[column])
            examples.append(row)
        return pd.DataFrame(examples)

    def get_llm_response(self, series, verbose=False):
        title, prompt = series['title'], series['full_prompt']
        tries = 0
        while True:
            tries += 1
            response = requests.post("http://localhost:11434/api/generate", json={"model": self.label_model, "prompt": prompt,
                    "system": self.task_instructions[self.task], "stream": False})
            result = response.json()["response"].strip()
            if result in self.task_pairings[self.task]:
                break
            if verbose:
                if tries > 10:
                    print(f'Over 10 tries to retrieve a valid response for {title[:60]}. Picking a label on random')
                    return random.choice(list(self.task_pairings[self.task].keys())), tries
                if tries > 5:
                    print(f'{tries} tries on labeling a specific entity: {title[:80]}...')
                    print(f'Model responded with: {result[:80]}...')
        return result, tries

    def get_llm_responses_parallel(self, df, verbose=False):
        start = time.time()
        tasks = [delayed(self.get_llm_response)(df.iloc[i], verbose) for i in range(len(df))]
        with ProgressBar():
            results = compute(*tasks)
            responses = [str.strip(response) for (response, tries) in results]
            for _, tries in results:
                self.attempts += 1
                self.failures += (tries - 1)
            print(f'Finished parallel LLM querying in {(time.time() - start) / 60}')
            return responses
