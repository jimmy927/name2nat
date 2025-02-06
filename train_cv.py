import os
import random
import argparse
from flair.data import Corpus, Sentence
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import OneHotEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from typing import List
import torch
from sklearn.model_selection import KFold
import numpy as np

# Reuse data preparation from original train.py
from train import convert

def create_fold_corpus(data_folder, column_name_map, train_indices, val_indices, train_data):
    """Create a corpus for a specific fold"""
    # Create temporary files for this fold
    os.makedirs(f"{data_folder}/fold", exist_ok=True)
    
    with open(f"{data_folder}/fold/train.txt", 'w', encoding='utf8') as f:
        for idx in train_indices:
            f.write(train_data[idx])
            
    with open(f"{data_folder}/fold/dev.txt", 'w', encoding='utf8') as f:
        for idx in val_indices:
            f.write(train_data[idx])
            
    return CSVClassificationCorpus(
        f"{data_folder}/fold",
        column_name_map,
        train_file="train.txt",
        dev_file="dev.txt",
        skip_header=False,
        delimiter='\t',
        label_type='label'
    )

def train_with_cv(k=5):
    # Prepare data
    convert('nana/train.src', 'nana/train.tgt', 'data/train.txt')
    convert('nana/dev.src', 'nana/dev.tgt', 'data/dev.txt')
    
    # Read all training data
    train_data = open('data/train.txt', 'r', encoding='utf8').readlines()
    
    # Setup k-fold cross validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    column_name_map = {0: "text", 1: "label"}
    scores = []
    best_score = 0
    best_model = None
    
    # Train k models
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
        print(f"\nTraining fold {fold+1}/{k}")
        
        # Create corpus for this fold
        corpus = create_fold_corpus('data', column_name_map, train_idx, val_idx, train_data)
        
        # Create model
        label_dict = corpus.make_label_dictionary(label_type='label')
        embeddings = [OneHotEmbeddings(vocab_dictionary=corpus.make_vocab_dictionary())]
        document_embeddings = DocumentRNNEmbeddings(embeddings, hidden_size=256, bidirectional=True)
        classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type='label')
        
        # Train
        trainer = ModelTrainer(classifier, corpus)
        trainer.train(
            f'resources/fold_{fold}',
            learning_rate=0.1,
            mini_batch_size=128,
            max_epochs=20,
            anneal_factor=0.5,
            patience=5,
            min_learning_rate=0.0001,
            train_with_dev=False,
            shuffle=True
        )
        
        # Evaluate on validation set
        score = classifier.evaluate(corpus.dev)[0]  # Get main score
        scores.append(score)
        
        if score > best_score:
            best_score = score
            best_model = classifier
            best_model.save('resources/best-model.pt')
    
    print("\nCross-validation results (on training data):")
    print(f"CV Average score: {np.mean(scores):.3f} (±{np.std(scores):.3f})")
    print(f"CV Best score: {best_score:.3f}")
    
    # After CV, evaluate best model on original test set
    print("\nEvaluating best model on held-out test set...")
    test_corpus = CSVClassificationCorpus(
        'data',
        column_name_map,
        train_file=None,
        dev_file=None,
        test_file='test.txt',  # Original test set
        skip_header=False,
        delimiter='\t',
        label_type='label'
    )
    final_score = best_model.evaluate(test_corpus.test)[0]
    print(f"Final test score: {final_score:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    args = parser.parse_args()
    
    train_with_cv(k=args.folds) 