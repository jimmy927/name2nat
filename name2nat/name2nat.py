from name2nat.fix_path import fix_path

fix_path()
from flair.models import TextClassifier
from flair.data import Sentence
import os


class Name2nat:
    def __init__(self):
        # Load model
        self.classifier = TextClassifier.load(
            os.path.join(os.path.dirname(__file__), "best-model.pt")
        )

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

    def __call__(self, names, top_n=5):
        """
        Predict nationality for each name.
        Args:
            names: A list of names
            top_n: Number of predictions to return
        """
        if not isinstance(names, list):
            names = [names]

        results = []
        for name in names:
            # Convert name format
            name = name.replace(" ", "▁")
            name = " ".join(char for char in name)

            # Get model predictions
            sentence = Sentence(name)
            self.classifier.predict(sentence, return_probabilities_for_all_classes=True)
            probs = sentence.get_labels()
            # Get top N predictions
            top_preds = sorted(
                [(l.value, l.score) for l in probs], key=lambda x: x[1], reverse=True
            )[:top_n]
            results.append((name, top_preds))

        return results
