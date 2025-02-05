from flair.models import TextClassifier
from flair.data import Sentence
import pickle
import os
import torch
import warnings

CKPT = os.path.dirname(os.path.abspath(__file__)) + "/best-model.pt"
DICT = os.path.dirname(os.path.abspath(__file__)) + "/name2nats.pkl"

class CustomTextClassifier(TextClassifier):
    @classmethod
    def load(cls, model_path):
        # Suppress source change warnings
        warnings.filterwarnings('ignore', category=torch.serialization.SourceChangeWarning)
        
        try:
            # First try loading with default settings
            return super().load(model_path)
        except Exception as e:
            try:
                # If that fails, try with explicit torch loading
                with torch.set_grad_enabled(False):
                    state = torch.load(model_path, map_location='cpu')
                    model = cls._init_model_with_state_dict(state)
                    return model
            except Exception as e2:
                raise RuntimeError(f"Failed to load model. Try downgrading PyTorch to 2.0.1 and Flair to 0.12.2. Original error: {str(e2)}")

    @classmethod
    def _init_model_with_state_dict(cls, state):
        model = cls()
        model.load_state_dict(state)
        model.eval()
        return model

class Name2nat:
    def __init__(self, ckpt=CKPT, name2nats=DICT):
        try:
            self.classifier = CustomTextClassifier.load(ckpt)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Please ensure you have compatible versions: torch==2.0.1 and flair==0.12.2")
            raise
        self.name2nats = self.construct(name2nats)

    def construct(self, name2nats):
         return pickle.load(open(name2nats, 'rb'))

    def convert(self, name):
        name = name.replace(" ", "▁")
        name = " ".join(char for char in name)
        return name

    def restore(self, name):
        return name.replace(" ", "").replace("▁", " ")

    def get_top_n_results(self, sentence, top_n):
        results = sentence.labels
        results = [(each.value, each.score) for each in results]
        results = sorted(results, key=lambda x: x[1], reverse=True)
        results = results[:top_n]
        return results

    def __call__(self, names, top_n=1, use_dict=True, mini_batch_size=128):
        if not isinstance(names, list):
            names = [names]

        sentences = [Sentence(self.convert(name), use_tokenizer=True) for name in names]
        self.classifier.predict(sentences, mini_batch_size=mini_batch_size, multi_class_prob=True, verbose=len(sentences)>1000)

        ret = []
        for sent in sentences:
            name = self.restore(sent.to_tagged_string()) # plain string
            if use_dict:
                if name in self.name2nats:
                    results = [(nat, 1.0) for nat in self.name2nats[name]]
                else:
                    results = self.get_top_n_results(sent, top_n)
            else:
                results = self.get_top_n_results(sent, top_n)

            ret.append((name, results))
        return ret







