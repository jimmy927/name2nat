from name2nat import Name2nat
import torch

my_name2nat = Name2nat()

# test data
names = open("nana/test.src", 'r', encoding='utf8').read().splitlines()

with torch.no_grad():  # Added for efficiency
    try:
        results = my_name2nat(names, top_n=5, use_dict=False)
    except RuntimeError as e:
        print(f"Error during prediction: {e}")
        raise
    
with open("test.pred", "w", encoding="utf8") as fout:
    for r in results:
        preds = r[-1]
        preds = ",".join(each[0] for each in preds)
        fout.write(preds + "\n")
