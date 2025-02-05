from flair.models import TextClassifier
from flair.data import Sentence
import pickle
import os

CKPT = os.path.dirname(os.path.abspath(__file__)) + "/best-model.pt"
DICT = os.path.dirname(os.path.abspath(__file__)) + "/name2nats.pkl"

class Name2nat:
    def __init__(self, ckpt=CKPT, name2nats=DICT):
        try:
            self.classifier = TextClassifier.load(ckpt)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        
        try:
            self.name2nats = pickle.load(open(name2nats, 'rb'))
        except Exception as e:
            print(f"Error loading dictionary: {str(e)}")
            self.name2nats = {}

    def convert(self, name):
        name = name.replace(" ", "▁")
        name = " ".join(char for char in name)
        return name

    def restore(self, name):
        return name.replace(" ", "").replace("▁", " ")

    def get_top_n_results(self, sentence, top_n):
        results = [(label.value, label.score) for label in sentence.labels]
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_n]

    def __call__(self, names, top_n=1, use_dict=True, mini_batch_size=128):
        if not isinstance(names, list):
            names = [names]

        # Create sentences without tokenizer since we're doing character-level
        sentences = [Sentence(self.convert(name)) for name in names]
        
        # Predict with only supported parameters
        self.classifier.predict(sentences, mini_batch_size=mini_batch_size)

        ret = []
        for sent in sentences:
            name = self.restore(sent.text)
            if use_dict and name in self.name2nats:
                results = [(nat, 1.0) for nat in self.name2nats[name]]
            else:
                results = self.get_top_n_results(sent, top_n)
            ret.append((name, results))
        return ret







