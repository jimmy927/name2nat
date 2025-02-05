from name2nat import Name2nat
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Predict nationalities from names.")
    parser.add_argument(
        '--small',
        action='store_true',
        help='Use the debug model trained with --small flag.'
    )
    args = parser.parse_args()

    # Models are saved in resources/{prefix}/best-model.pt by Flair
    prefix = "debug-model" if args.small else "production-model"
    model_path = os.path.join("resources", prefix, "best-model.pt")  # Flair's actual model path
    
    my_name2nat = Name2nat(ckpt=model_path)

    # test data
    names = open("nana/test.src", 'r', encoding='utf8').read().splitlines()

    results = my_name2nat(names, top_n=5, use_dict=False) # use_dict: dictionary retrieval. we don't want it for model prediction.
    with open("test.pred", "w", encoding="utf8") as fout:
        for r in results:
            preds = r[-1]
            preds = ",".join(each[0] for each in preds)
            fout.write(preds + "\n")

if __name__ == "__main__":
    main()
