from fix_path import fix_path
fix_path()

import sys
import os
import country_converter as coco
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from name2nat.name2nat import Name2nat

def get_country_name(code: str) -> str:
    """Convert ISO2 country code to full name"""
    cc = coco.CountryConverter()
    try:
        return cc.convert(names=code, to='name_short')
    except:
        return code.upper()

my_nanat = Name2nat()

names = [
    "Donald Trump",  # American
    "Moon Jae-in",  # Korean
    "Shinzo Abe",  # Japanese
    "Xi Jinping",  # Chinese
    "Joko Widodo",  # Indonesian
    "Angela Merkel",  # German
    "Emmanuel Macron",  # French
    "Kyubyong Park",  # Korean
    "Yamamoto Yu",  # Japanese
    "Jing Xu",  # Chinese
    "Mikael Andersson",  # Swedish
    "Jimmy Engelbrecht",
    "Maryam Engelbrecht",
    "Kyubyong Park",
    "Sofia Jonsson",
    "Axel Petterson",
    "Hassan Hamid",
    "Kasper Petersen",
    "Senja Hansen",
    "Dennis Andersen",
    "Pekka Nieminen",
    "Joonas Kinnunen",
    "Marcus Johansson",
    "Erik Olssen",
    "Caspian Shahrami",
    "Jesper Nilsson",
    "Jesper Olsson",
    "Jesper Eriksson",
    "Jesper Hansen",
    "Jesper Andersen",
    "Jesper Jensen",
    "Caspian Andersson",
    "Caspian Hadid",
    "Ali Andersson",
    "Ali Hadid",
    "Omar Al-Masri",
    "Fatima Al-Khatib",
    "Mahmoud Haddad",
    "Amina Al-Assad",
    "Khaled Dabbagh",
    "Rania Al-Khalil",
    "Youssef Barakat",
    "Layla Al-Jabri",
    "Tariq Al-Hamwi",
    "Nour Suleiman",
    "Joseph Stalin",
    "Vladimir Lenin",
    "Mao Zedong",
    "Kim Il-sung",
    "Kim Jong-il",
    "Kim Jong-un",
    "Vladimir Putin",
    "Xi Jinping",
    "Pol Pot",
    "Svenne Banan"
]


results = my_nanat(names)
print("\nPredicted Nationalities:")
print("-" * 50)
for i, (processed_name, predictions) in enumerate(results):
    print(f"\n{names[i]}:")
    # Sort predictions by probability
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    for nat, prob in predictions:
        if prob < 0.1:  # Skip very low probabilities
            continue
        country = get_country_name(nat)
        print(f"  {country:15} {prob*100:4.1f}%") 