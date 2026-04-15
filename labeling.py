import torch
import os
import pandas as pd
import requests

from pprint import pprint
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


class Labeling:
    def __init__(self, label_model= "huggingface"):
        self.label_model = label_model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    def generate_prompt(self, title):
        if self.label_model == "huggingface": return self.generate_prompt(title)
        elif self.label_model=="gpt": return self.generate_prompt(title)
        else: return None

    def generate_prompt(self, title):
        return f"""
            You are labeling tool to create labels for a classification task.
            I will provide text data from an advertisement of a product. The product should be classified in two labels:
            
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

    def set_model(self):
        if self.label_model == "huggingface":
            # checkpoint = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            # self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(self.device)
            # self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            pass
        elif self.label_model == "gpt": self.model = OpenAI(api_key="YOUR_OPENAI_API_KEY")
        else: raise ValueError("No such model")

    def predict_animal_product(self, row):
        if self.label_model == "huggingface": return self.get_huggingface_label(row)
        elif self.label_model == "gpt": return self.get_gpt_label(row)
        else: raise ValueError("No model selected")

    def generate_inference_data(self, data, column):
        examples = []
        for _, data_point in data.iterrows():
            examples.append({
                "id": data_point["id"],
                "title": data_point["title"],
                "training_text": data_point["clean_title"],
                "text": self.generate_prompt(data_point[column]),
            })
        data = pd.DataFrame(examples)
        return data

    def get_gpt_label(self, row):
        id_, prompt = row["id"], row["text"]
        response = self.model.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt},],
            max_tokens=100,
            temperature=0.2,
        )
        return response.choices[0].message.content

    def get_huggingface_label(self, row):
        # using Ollama Llama 3; download it and then pull it
        id_, prompt = row["id"], row["text"]
        tries = 0
        while True:
            response = requests.post("http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": prompt,
                    "suffix": "",  
                    "system": "You are a labeling tool. You must only respond with 'relevant animal' or \
                            'not a relevant animal'. Never ask questions or explain yourself.",
                    "stream": False
                })
            result = response.json()["response"]
            if 'not a relevant animal' in result or 'relevant animal' in result: break
            tries += 1
            if tries > 2: print(f'{tries} tries on labeling a specific advertisement.')
        return result
