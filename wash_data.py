import os
import re
import langcodes
from tqdm import tqdm
import country_converter as coco
from babel import Locale
cc = coco.CountryConverter()

def clean_name(name: str) -> str:
    """Clean a name by stripping whitespace and removing special characters"""
    name = name.strip()
    
    # Remove emojis and special characters
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    name = emoji_pattern.sub("", name).strip()
    return name

def capitalize_name(name: str) -> str:
    """Properly capitalize a name"""
    name = name.lower()
    
    # Handle hyphenated names
    if "-" in name:
        return "-".join(word.capitalize() for word in name.split("-"))
    
    # Known lowercase words
    lowercase_words = {
        'von', 'van', 'der', 'de', 'la', 'le', 'di', 'da', 'del',
        'dos', 'das', 'do', 'jr', 'sr'
    }
    
    words = name.split()
    if not words:
        return name
        
    result = [words[0].capitalize()]
    result.extend(word if word in lowercase_words else word.capitalize() 
                 for word in words[1:])
    
    return " ".join(result)

def convert_to_language_code(nationality: str) -> str:
    """Convert nationality to ISO 639-1 language code"""
    language_map = {
        "afghan": "ps",  # Pashto
        "albanian": "sq",
        "algerian": "ar",
        "american": "en",
        "andorran": "ca",  # Catalan
        "angolan": "pt",
        "argentine": "es",
        "armenian": "hy",
        "aruban": "nl",
        "australian": "en",
        "austrian": "de",
        "azerbaijani": "az",
        "bahamian": "en",
        "bahraini": "ar",
        "bangladeshi": "bn",
        "barbadian": "en",
        "basque": "eu",
        "belarusian": "be",
        "belgian": "nl",  # Primary language Nederlands (could also be "fr" French)
        "belizean": "en",
        "beninese": "fr",
        "bermudian": "en",
        "bhutanese": "dz",  # Dzongkha
        "bolivian": "es",
        "bosniak": "bs",
        "botswana": "en",
        "brazilian": "pt",
        "breton": "br",
        "british": "en",
        "bruneian": "ms",  # Malay
        "bulgarian": "bg",
        "burkinabé": "fr",
        "burmese": "my",
        "burundian": "rn",  # Kirundi
        "cambodian": "km",
        "cameroonian": "fr",  # Also en
        "canadian": "en",  # Also fr
        "catalan": "ca",
        "chadian": "ar",  # Also fr
        "chilean": "es",
        "chinese": "zh",
        "colombian": "es",
        "comorian": "ar",
        "congolese": "fr",
        "cuban": "es",
        "cypriot": "el",  # Greek
        "czech": "cs",
        "dane": "da",
        "djiboutian": "ar",
        "dominican": "es",
        "dutch": "nl",      # Nederlands
        "ecuadorian": "es",
        "egyptian": "ar",
        "emirati": "ar",
        "english": "en",
        "equatoguinean": "es",
        "eritrean": "ti",  # Tigrinya
        "estonian": "et",
        "ethiopian": "am",  # Amharic
        "faroese": "fo",
        "filipino": "tl",  # Tagalog
        "finn": "fi",
        "french": "fr",
        "gabonese": "fr",
        "gambian": "en",
        "georgian": "ka",
        "german": "de",
        "ghanaian": "en",
        "gibraltarian": "en",
        "greek": "el",
        "grenadian": "en",
        "guatemalan": "es",
        "guinean": "fr",
        "guyanese": "en",
        "haitian": "ht",  # Haitian Creole
        "honduran": "es",
        "hungarian": "hu",
        "i-kiribati": "en",
        "indian": "hi",  # Hindi (many others)
        "indonesian": "id",
        "iranian": "fa",  # Persian
        "iraqi": "ar",
        "irish": "ga",  # Irish Gaelic
        "israeli": "he",  # Hebrew
        "italian": "it",
        "jamaican": "en",
        "japanese": "ja",
        "jordanian": "ar",
        "kazakh": "kk",
        "kenyan": "sw",  # Swahili
        "korean": "ko",
        "kuwaiti": "ar",
        "kyrgyz": "ky",
        "lao": "lo",
        "latvian": "lv",
        "lebanese": "ar",
        "liberian": "en",
        "libyan": "ar",
        "lithuanian": "lt",
        "macedonian": "mk",
        "malagasy": "mg",
        "malawian": "ny",  # Chichewa
        "malaysian": "ms",
        "maldivian": "dv",  # Divehi
        "malian": "fr",
        "maltese": "mt",
        "manx": "gv",  # Manx Gaelic
        "marshallese": "mh",
        "mauritanian": "ar",
        "mauritian": "fr",
        "mexican": "es",
        "moldovan": "ro",
        "mongolian": "mn",
        "montenegrin": "sr",  # Serbian
        "moroccan": "ar",
        "mozambican": "pt",
        "namibian": "en",
        "nauruan": "na",
        "nepalese": "ne",
        "nicaraguan": "es",
        "nigerian": "en",
        "nigerien": "fr",
        "norwegian": "no",
        "omani": "ar",
        "pakistani": "ur",  # Urdu
        "palauan": "pau",
        "palestinian": "ar",
        "panamanian": "es",
        "paraguayan": "es",
        "peruvian": "es",
        "portuguese": "pt",
        "qatari": "ar",
        "romanian": "ro",
        "russian": "ru",
        "rwandan": "rw",  # Kinyarwanda
        "salvadoran": "es",
        "sammarinese": "it",
        "samoan": "sm",
        "saudi": "ar",
        "senegalese": "fr",
        "serb": "sr",
        "singaporean": "en",  # Also zh, ms, ta
        "slovak": "sk",
        "slovene": "sl",
        "somali": "so",
        "sotho": "st",  # Southern Sotho
        "sudanese": "ar",
        "surinamese": "nl", # Primary language Nederlands
        "swazi": "ss",  # Swati
        "syriac": "syr",
        "syrian": "ar",
        "taiwanese": "zh",
        "tajik": "tg",
        "tamil": "ta",
        "tanzanian": "sw",  # Swahili
        "thai": "th",
        "tibetan": "bo",
        "togolese": "fr",
        "tongan": "to",
        "tunisian": "ar",
        "turk": "tr",
        "tuvaluan": "tvl",
        "ugandan": "en",
        "ukrainian": "uk",
        "uruguayan": "es",
        "uzbek": "uz",
        "vanuatuan": "bi",  # Bislama
        "venezuelan": "es",
        "vietnamese": "vi",
        "vincentian": "en",
        "welsh": "cy",
        "yemeni": "ar",
        "zambian": "en"
    }
    return language_map.get(nationality.lower())

def convert_to_country_code(nationality: str) -> str:
    """Convert nationality to ISO 3166-1 alpha-2 country code"""
    country_map = {
        "afghan": "af",
        "albanian": "al",
        "algerian": "dz",
        "american": "us",
        "andorran": "ad",
        "angolan": "ao",
        "argentine": "ar",
        "armenian": "am",
        "aruban": "aw",
        "australian": "au",
        "austrian": "at",
        "azerbaijani": "az",
        "bahamian": "bs",
        "bahraini": "bh",
        "bangladeshi": "bd",
        "barbadian": "bb",
        "basque": "es",
        "belarusian": "by",
        "belgian": "be",
        "belizean": "bz",
        "beninese": "bj",
        "bermudian": "bm",
        "bhutanese": "bt",
        "bolivian": "bo",
        "bosniak": "ba",
        "botswana": "bw",
        "brazilian": "br",
        "breton": "fr",
        "british": "gb",
        "bruneian": "bn",
        "bulgarian": "bg",
        "burkinabé": "bf",
        "burmese": "mm",
        "burundian": "bi",
        "cambodian": "kh",
        "cameroonian": "cm",
        "canadian": "ca",
        "catalan": "es",
        "chadian": "td",
        "chilean": "cl",
        "chinese": "cn",
        "colombian": "co",
        "comorian": "km",
        "congolese": "cd",
        "cuban": "cu",
        "cypriot": "cy",
        "czech": "cz",
        "dane": "dk",
        "djiboutian": "dj",
        "dominican": "do",
        "dutch": "nl",
        "ecuadorian": "ec",
        "egyptian": "eg",
        "emirati": "ae",
        "english": "gb",
        "equatoguinean": "gq",
        "eritrean": "er",
        "estonian": "ee",
        "ethiopian": "et",
        "faroese": "fo",
        "filipino": "ph",
        "finn": "fi",
        "french": "fr",
        "gabonese": "ga",
        "gambian": "gm",
        "georgian": "ge",
        "german": "de",
        "ghanaian": "gh",
        "gibraltarian": "gi",
        "greek": "gr",
        "grenadian": "gd",
        "guatemalan": "gt",
        "guinean": "gn",
        "guyanese": "gy",
        "haitian": "ht",
        "honduran": "hn",
        "hungarian": "hu",
        "i-kiribati": "ki",
        "indian": "in",
        "indonesian": "id",
        "iranian": "ir",
        "iraqi": "iq",
        "irish": "ie",
        "israeli": "il",
        "italian": "it",
        "jamaican": "jm",
        "japanese": "jp",
        "jordanian": "jo",
        "kazakh": "kz",
        "kenyan": "ke",
        "korean": "kr",
        "kuwaiti": "kw",
        "kyrgyz": "kg",
        "lao": "la",
        "latvian": "lv",
        "lebanese": "lb",
        "liberian": "lr",
        "libyan": "ly",
        "lithuanian": "lt",
        "macedonian": "mk",
        "malagasy": "mg",
        "malawian": "mw",
        "malaysian": "my",
        "maldivian": "mv",
        "malian": "ml",
        "maltese": "mt",
        "manx": "gb",
        "marshallese": "mh",
        "mauritanian": "mr",
        "mauritian": "mu",
        "mexican": "mx",
        "moldovan": "md",
        "mongolian": "mn",
        "montenegrin": "me",
        "moroccan": "ma",
        "mozambican": "mz",
        "namibian": "na",
        "nauruan": "nr",
        "nepalese": "np",
        "nicaraguan": "ni",
        "nigerian": "ng",
        "nigerien": "ne",
        "norwegian": "no",
        "omani": "om",
        "pakistani": "pk",
        "palauan": "pw",
        "palestinian": "ps",
        "panamanian": "pa",
        "paraguayan": "py",
        "peruvian": "pe",
        "portuguese": "pt",
        "qatari": "qa",
        "romanian": "ro",
        "russian": "ru",
        "rwandan": "rw",
        "salvadoran": "sv",
        "sammarinese": "sm",
        "samoan": "ws",
        "saudi": "sa",
        "senegalese": "sn",
        "serb": "rs",
        "singaporean": "sg",
        "slovak": "sk",
        "slovene": "si",
        "somali": "so",
        "sotho": "za",
        "sudanese": "sd",
        "surinamese": "sr",
        "swazi": "sz",
        "syriac": "sy",
        "syrian": "sy",
        "taiwanese": "tw",
        "tajik": "tj",
        "tamil": "lk",
        "tanzanian": "tz",
        "thai": "th",
        "tibetan": "cn",
        "togolese": "tg",
        "tongan": "to",
        "tunisian": "tn",
        "turk": "tr",
        "tuvaluan": "tv",
        "ugandan": "ug",
        "ukrainian": "ua",
        "uruguayan": "uy",
        "uzbek": "uz",
        "vanuatuan": "vu",
        "venezuelan": "ve",
        "vietnamese": "vn",
        "vincentian": "vc",
        "welsh": "gb",
        "yemeni": "ye",
        "zambian": "zm"
    }
    return country_map.get(nationality.lower())

def process_files(input_dir: str, output_dir: str):
    """Process all files and create both language and country code versions"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "lang"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "country"), exist_ok=True)
    
    # Track unique conversions
    seen_nationalities = {}
    
    for split in ['train', 'dev', 'test']:
        src_file = os.path.join(input_dir, f'{split}.src')
        tgt_file = os.path.join(input_dir, f'{split}.tgt')
        
        if not (os.path.exists(src_file) and os.path.exists(tgt_file)):
            continue
            
        print(f"\nProcessing {split} files...")
        
        with open(src_file, 'r', encoding='utf8') as f:
            names = f.read().strip().splitlines()
        with open(tgt_file, 'r', encoding='utf8') as f:
            nationalities = f.read().strip().splitlines()
            
        # Process names and both types of codes
        cleaned_data = []
        for name, nat in tqdm(zip(names, nationalities), total=len(names)):
            cleaned_name = capitalize_name(clean_name(name))
            lang_code = convert_to_language_code(nat)
            country_code = convert_to_country_code(nat)
            
            # Print new conversions
            if nat not in seen_nationalities:
                print(f"Converting {nat} -> lang: {lang_code}, country: {country_code}")
                seen_nationalities[nat] = (lang_code, country_code)
            
            if cleaned_name and lang_code and country_code:
                cleaned_data.append((cleaned_name, lang_code, country_code))
        
        # Write files
        with open(os.path.join(output_dir, "lang", f'{split}.src'), 'w', encoding='utf8') as f:
            f.write('\n'.join(item[0] for item in cleaned_data))
        with open(os.path.join(output_dir, "lang", f'{split}.tgt'), 'w', encoding='utf8') as f:
            f.write('\n'.join(item[1] for item in cleaned_data))
            
        with open(os.path.join(output_dir, "country", f'{split}.src'), 'w', encoding='utf8') as f:
            f.write('\n'.join(item[0] for item in cleaned_data))
        with open(os.path.join(output_dir, "country", f'{split}.tgt'), 'w', encoding='utf8') as f:
            f.write('\n'.join(item[2] for item in cleaned_data))
            
        print(f"Processed {len(cleaned_data)} entries for {split}")

if __name__ == "__main__":
    input_dir = "nana"
    output_dir = "nana_clean"
    
    print("Starting data cleaning process...")
    process_files(input_dir, output_dir)
    print("\nDone! Cleaned files are in the nana_clean directory") 