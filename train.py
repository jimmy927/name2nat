import os
import random
import argparse
from typing import Dict, List
from flair.data import Corpus, Dictionary, Sentence, FlairDataset
from flair.datasets import CSVClassificationCorpus, ClassificationDataset
from flair.embeddings import OneHotEmbeddings, DocumentRNNEmbeddings
from flair.embeddings.token import StackedEmbeddings
from flair.models import TextClassifier as FlairTextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path
import torch
from torch.serialization import add_safe_globals
from torch.nn import (
    Embedding, Linear, LSTM, GRU, RNN,
    Dropout, ReLU, Module,
    ModuleList, Parameter, Sequential
)

# Add required classes to safe globals for PyTorch 2.6+
SAFE_CLASSES = [
    DocumentRNNEmbeddings,
    StackedEmbeddings,
    OneHotEmbeddings,
    Embedding,
    torch.nn.modules.sparse.Embedding,
    Dictionary,
    Linear,
    torch.nn.modules.linear.Linear,
    LSTM,
    torch.nn.modules.rnn.LSTM,
    GRU,
    torch.nn.modules.rnn.GRU,
    RNN,
    torch.nn.modules.rnn.RNN,
    Dropout,
    torch.nn.modules.dropout.Dropout,
    ReLU,
    torch.nn.modules.activation.ReLU,
    Module,
    torch.nn.modules.module.Module,
    ModuleList,
    torch.nn.modules.container.ModuleList,
    Parameter,
    torch.nn.parameter.Parameter,
    Sequential,
    torch.nn.modules.container.Sequential
]
add_safe_globals(SAFE_CLASSES)

def convert(name_f, nat_f, fout):
    with open(fout, 'w', encoding='utf8') as fout:
        names = open(name_f, 'r', encoding='utf8').read().strip().splitlines()
        nats = open(nat_f, 'r', encoding='utf8').read().strip().splitlines()
        for name, nat in zip(names, nats):
            if "train" in name_f and nat == "Korean":
                if random.random() > 0.5:
                    name = name.replace("-", "")
                if random.random() > 0.5:
                    columns = name.split(" ", 1)
                    if len(columns)==2:
                        last, first = columns
                        name = first + " " + last
            name = name.replace(" ", "▁")
            name = " ".join(char for char in name)
            fout.write(f"{name}\t{nat}\n")

def create_char_dictionary(corpus: Corpus) -> Dictionary:
    """Create a dictionary of all characters in the corpus."""
    char_dict = Dictionary()
    # Add special tokens
    char_dict.add_item('<unk>')
    char_dict.add_item(' ')
    
    # Add all characters from the training data
    for sentence in corpus.train:
        for char in sentence.to_plain_string():
            char_dict.add_item(char)
    
    return char_dict

def check_file_exists(filepath: str, description: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find {description} file at {filepath}")

def verify_data_format(filepath: str):
    """Verify that the data file is properly formatted."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                raise ValueError(f"Invalid format in {filepath} at line {i}. Expected 2 tab-separated columns, got {len(parts)}")
            text, label = parts
            if not text or not label:
                raise ValueError(f"Empty text or label in {filepath} at line {i}")

def load_classification_data(file_path: str, label_type: str = 'nationality') -> List[Sentence]:
    """Load data from a tab-separated file into a list of labeled sentences."""
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text, label = line.strip().split('\t')
            sentence = Sentence(text)
            sentence.add_label(label_type, label)
            sentences.append(sentence)
    return sentences

def downsample_data(lines: List[str], max_samples: int = 500) -> List[str]:
    """Downsample data while maintaining nationality distribution."""
    # Group lines by nationality
    nationality_groups = {}
    for line in lines:
        name, nat = line.strip().split('\t')
        if nat not in nationality_groups:
            nationality_groups[nat] = []
        nationality_groups[nat].append(line)
    
    # Sample from each nationality group
    sampled_lines = []
    samples_per_nat = max(1, max_samples // len(nationality_groups))  # At least 1 sample per nationality
    for nat, nat_lines in nationality_groups.items():
        if len(nat_lines) > samples_per_nat:
            sampled_lines.extend(random.sample(nat_lines, samples_per_nat))
        else:
            sampled_lines.extend(nat_lines)
    
    # Shuffle the sampled lines
    random.shuffle(sampled_lines)
    
    # Cap at max_samples if we have more
    if len(sampled_lines) > max_samples:
        sampled_lines = sampled_lines[:max_samples]
    
    return sampled_lines

class CustomTextClassifier(FlairTextClassifier):
    @classmethod
    def load(cls, model_path):
        """Custom load method that handles PyTorch 2.6+ loading behavior."""
        try:
            # First try loading with weights_only=False
            with torch.serialization.safe_globals(SAFE_CLASSES):
                state = torch.load(str(model_path), map_location='cpu', weights_only=False)
            
            # Create a new model instance with the saved parameters
            model = cls(
                embeddings=state['embeddings'],
                label_dictionary=state['label_dictionary'],
                label_type=state['label_type']
            )
            
            # Load the state dict
            if 'state_dict' in state:
                model.load_state_dict(state['state_dict'])
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            try:
                # Try loading with base class method as fallback
                with torch.serialization.safe_globals(SAFE_CLASSES):
                    return super().load(model_path)
            except Exception as e2:
                print(f"Fallback loading also failed: {str(e2)}")
                raise

    def save(self, model_file: str, checkpoint: bool = False):
        """Custom save method to ensure all necessary attributes are saved.
        
        Args:
            model_file: Path to save the model to
            checkpoint: If True, also saves training state for resuming training
        """
        # Save the full model state
        model_state = {
            'state_dict': self.state_dict(),
            'embeddings': self.embeddings,  # Save the entire embeddings object
            'label_dictionary': self.label_dictionary,
            'label_type': self.label_type
        }
        
        # If saving checkpoint, include optimizer state
        if checkpoint:
            model_state['optimizer_state_dict'] = self.optimizer.state_dict() if hasattr(self, 'optimizer') else None
            model_state['scheduler_state_dict'] = self.scheduler.state_dict() if hasattr(self, 'scheduler') else None
            
        # Save with safe globals context
        with torch.serialization.safe_globals(SAFE_CLASSES):
            torch.save(model_state, str(model_file))

def main():
    parser = argparse.ArgumentParser(description="Train the Name2nat model.")
    parser.add_argument(
        '--small',
        action='store_true',
        help='Train with a small subset of data for debugging.'
    )
    args = parser.parse_args()

    # Check source files exist
    check_file_exists('nana/train.src', 'training source')
    check_file_exists('nana/train.tgt', 'training target')
    check_file_exists('nana/dev.src', 'development source')
    check_file_exists('nana/dev.tgt', 'development target')

    # Create data directory
    os.makedirs('data', exist_ok=True)

    # Convert data files
    convert('nana/train.src', 'nana/train.tgt', 'data/train.txt')
    convert('nana/dev.src', 'nana/dev.tgt', 'data/dev.txt')

    # Verify converted files exist and have content
    check_file_exists('data/train.txt', 'converted training')
    check_file_exists('data/dev.txt', 'converted development')

    # Verify data format
    print("Verifying data format...")
    verify_data_format('data/train.txt')
    verify_data_format('data/dev.txt')
    print("Data format verification successful")

    # Load corpus
    data_folder = os.path.abspath('data')  # Use absolute path
    column_name_map = {0: "text", 1: "nationality"}
    
    # If in debug mode, downsample the data before loading
    if args.small:
        print("Debug mode activated: using a small subset of the data.")
        
        # Read and downsample training data
        with open('data/train.txt', 'r', encoding='utf8') as f:
            train_lines = f.readlines()
        
        if not train_lines:
            raise ValueError("Training file is empty!")
            
        sampled_train = downsample_data(train_lines, max_samples=500)
        print(f"Sampled {len(sampled_train)} training examples")
        
        # Read and downsample dev data
        with open('data/dev.txt', 'r', encoding='utf8') as f:
            dev_lines = f.readlines()
            
        if not dev_lines:
            raise ValueError("Dev file is empty!")
            
        sampled_dev = downsample_data(dev_lines, max_samples=100)  # Smaller dev set
        print(f"Sampled {len(sampled_dev)} dev examples")
        
        # Write downsampled data to new files
        with open('data/train_small.txt', 'w', encoding='utf8') as f:
            f.writelines(sampled_train)
            
        with open('data/dev_small.txt', 'w', encoding='utf8') as f:
            f.writelines(sampled_dev)
        
        train_file = "train_small.txt"
        dev_file = "dev_small.txt"
    else:
        train_file = "train.txt"
        dev_file = "dev.txt"
    
    # Verify the files we're about to use exist
    train_path = os.path.join(data_folder, train_file)
    dev_path = os.path.join(data_folder, dev_file)
    
    print(f"Loading corpus from:")
    print(f"- Training file: {train_path}")
    print(f"- Dev file: {dev_path}")
    
    check_file_exists(train_path, 'training')
    check_file_exists(dev_path, 'development')
    
    # Create corpus by loading datasets directly
    train_data = load_classification_data(train_path)
    dev_data = load_classification_data(dev_path)
    
    # Create a Corpus with our lists of sentences
    class SentenceDataset(FlairDataset):
        def __init__(self, sentences: List[Sentence]):
            self.sentences = sentences

        def __len__(self):
            return len(self.sentences)

        def __getitem__(self, index: int) -> Sentence:
            return self.sentences[index]

    corpus = Corpus(
        train=SentenceDataset(train_data),
        dev=SentenceDataset(dev_data),
        test=None
    )

    # Verify corpus loaded successfully
    if not corpus.train or len(corpus.train) == 0:
        raise ValueError("No training examples found in corpus!")
    if not corpus.dev or len(corpus.dev) == 0:
        raise ValueError("No development examples found in corpus!")

    print(f"Successfully loaded {len(corpus.train)} training examples and {len(corpus.dev)} development examples")

    # Print corpus statistics
    stats = corpus.obtain_statistics()
    print(stats)

    # Create label dictionary
    label_dict = corpus.make_label_dictionary(label_type='nationality')
    print("Label dictionary:", label_dict)
    print("Number of labels:", len(label_dict))

    # Create character dictionary and embeddings
    char_dict = create_char_dictionary(corpus)
    print(f"Created character dictionary with {len(char_dict)} characters")
    
    # Create embeddings with the character dictionary
    embeddings = [OneHotEmbeddings(vocab_dictionary=char_dict)]
    document_embeddings = DocumentRNNEmbeddings(embeddings, bidirectional=True, hidden_size=256)
    classifier = CustomTextClassifier(
        document_embeddings,
        label_dictionary=label_dict,
        label_type='nationality'
    )

    # Initialize trainer and start training
    trainer = ModelTrainer(classifier, corpus)
    trainer.train('resources/',
                 learning_rate=0.1,
                 mini_batch_size=128,
                 anneal_factor=0.5,
                 patience=5,
                 max_epochs=20)

if __name__ == '__main__':
    main()