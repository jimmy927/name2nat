import name2nat
from name2nat import Name2nat
import torch
import os
import inspect
from datetime import datetime

# Print where Name2nat is looking for the model
print("Name2nat package location:", os.path.dirname(name2nat.__file__))

model_path = "resources/best-model.pt"
if not os.path.exists(model_path):
    print(f"Error: No model found at {model_path}")
    print("Please train a new model first using train.py")
    exit(1)

# Print model paths and timestamps
print(f"New model location: {os.path.abspath(model_path)}")
print(f"New model timestamp: {datetime.fromtimestamp(os.path.getmtime(model_path))}")

# Try to read model metadata
try:
    model_data = torch.load(model_path, map_location='cpu')
    if hasattr(model_data, 'model_card'):
        print("\nModel metadata:")
        print(f"Flair version: {model_data.model_card.get('flair_version', 'unknown')}")
        print(f"PyTorch version: {model_data.model_card.get('pytorch_version', 'unknown')}")
except Exception as e:
    print(f"\nCould not read model metadata: {e}")

name2nat_init = inspect.getsourcefile(Name2nat.__init__)
print(f"\nName2nat __init__ location: {name2nat_init}")
if os.path.exists(name2nat_init):
    print(f"Name2nat __init__ timestamp: {datetime.fromtimestamp(os.path.getmtime(name2nat_init))}")

# Copy or move the new model to where Name2nat expects it
print(f"Found model at: {model_path}")

my_name2nat = Name2nat()

# test data
names = open("nana/test.src", 'r', encoding='utf8').read().splitlines()

with torch.no_grad():
    try:
        results = my_name2nat(names, top_n=5, use_dict=False)
    except RuntimeError as e:
        print(f"Error during prediction: {e}")
        print("Unexpected error with PyTorch 2.5 model")
        raise
    
with open("test.pred", "w", encoding="utf8") as fout:
    for r in results:
        preds = r[-1]
        preds = ",".join(each[0] for each in preds)
        fout.write(preds + "\n")
