import os
import random
import argparse
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import OneHotEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List
import torch
import flair

# Update argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--small', action='store_true', help='Use only 0.1%% of samples for training')
parser.add_argument('--sample-pct', type=float, default=100.0, 
                   help='Percentage of training data to use (0.1-100.0). Default: 100.0')
parser.add_argument('--sample-dev', action='store_true', 
                   help='Apply sampling to dev set as well (not recommended)')
parser.add_argument('--max-epochs', type=int, default=20,
                   help='Maximum number of epochs to train. Default: 20')
parser.add_argument('--mini-batch-size', type=int, default=128,
                   help='Size of mini-batches during training. Default: 128. '
                        'Larger values use more memory but train faster. '
                        'Reduce this if you get out-of-memory errors.')
args = parser.parse_args()

os.makedirs('data', exist_ok=True)

def convert(name_f, nat_f, fout, sample_percentage=100.0, sample_dev=False):
    with open(fout, 'w', encoding='utf8') as fout:
        names = open(name_f, 'r', encoding='utf8').read().strip().splitlines()
        nats = open(nat_f, 'r', encoding='utf8').read().strip().splitlines()
        
        print(f"\nProcessing {name_f}:")
        print(f"Total samples available: {len(names)}")
        
        # If sampling is requested (and it's either train or we want to sample dev too)
        if sample_percentage < 100.0 and ('train' in name_f or sample_dev):
            combined = list(zip(names, nats))
            sample_size = max(int(len(combined) * sample_percentage / 100.0), 1)
            combined = random.sample(combined, sample_size)
            names, nats = zip(*combined)
            print(f"Using {sample_percentage}% of data: selected {len(names)} samples")
        else:
            print(f"Using all {len(names)} samples")
        
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

# Update convert calls to use either --small or --sample-pct
sample_pct = 0.1 if args.small else args.sample_pct
convert('nana/train.src', 'nana/train.tgt', 'data/train.txt', 
        sample_percentage=sample_pct, sample_dev=args.sample_dev)
convert('nana/dev.src', 'nana/dev.tgt', 'data/dev.txt', 
        sample_percentage=sample_pct if args.sample_dev else 100.0)

# this is the folder in which train, test and dev files reside
data_folder = 'data'

# column format indicating which columns hold the text and label(s)
column_name_map = {0: "text", 1: "label"}

# load corpus containing training, test and dev data
corpus: Corpus = CSVClassificationCorpus(
    data_folder,
    column_name_map,
    train_file="train.txt",
    dev_file="dev.txt",
    skip_header=False,
    delimiter='\t',    # tab-separated files
    label_type='label'  # Added label_type parameter
)

stats = corpus.obtain_statistics()
print(stats)

# create the label dictionary from all data
label_dict = corpus.make_label_dictionary(label_type='label', add_dev_test=True)
print(label_dict)

# make a list of word embeddings
embeddings: List[OneHotEmbeddings] = [OneHotEmbeddings(
    vocab_dictionary=corpus.make_vocab_dictionary()
)]

# initialize document embedding by passing list of word embeddings
# Can choose between many RNN types (GRU by default, to change use rnn_type parameter)
document_embeddings = DocumentRNNEmbeddings(
    embeddings, 
    hidden_size=256,
    bidirectional=True
)

# create the text classifier
classifier = TextClassifier(
    document_embeddings,
    label_dictionary=label_dict,
    label_type='label'  # Add label_type parameter to match what we used in corpus
)

# initialize the text classifier trainer
trainer = ModelTrainer(classifier, corpus)

# start the training
trainer.train(
    'resources/',
    learning_rate=0.1,
    mini_batch_size=args.mini_batch_size,
    max_epochs=args.max_epochs,
    anneal_factor=0.5,
    patience=5,
    min_learning_rate=0.0001,
    train_with_dev=False,
    shuffle=True
)