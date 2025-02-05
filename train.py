import os
import random
import argparse
from typing import List
from flair.data import Corpus, Dictionary, Sentence
from flair.embeddings import OneHotEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

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
    
    # Create corpus using standard Flair Corpus
    corpus = Corpus(train=train_data, dev=dev_data, test=[])

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
    document_embeddings = DocumentRNNEmbeddings(
        embeddings=embeddings,
        hidden_size=256,
        rnn_type='GRU',
        reproject_words=True,
        reproject_words_dimension=256,
        bidirectional=True,
    )
    
    classifier = TextClassifier(
        document_embeddings,
        label_dictionary=label_dict,
        label_type='nationality'
    )

    # Initialize trainer and start training
    trainer = ModelTrainer(classifier, corpus)
    
    # All models are saved in resources/ with clear naming:
    # - {prefix}-best.pt: Model with best validation performance during training
    # - {prefix}-last.pt: Model from the last training epoch
    model_prefix = "debug-model" if args.small else "production-model"
    
    # Training parameters - only using parameters supported by Flair 0.15.0
    trainer.train(
        base_path=os.path.join('resources', model_prefix),
        learning_rate=0.1,
        mini_batch_size=128,
        anneal_factor=0.5,
        patience=5,
        max_epochs=20
    )

if __name__ == '__main__':
    main()