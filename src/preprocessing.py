"""
Text Preprocessing Module for Indic Languages (Hindi & Bengali)
Includes Unicode normalization, noise removal, Indic stopword handling, and tokenization.
"""

import os
import re
import unicodedata
import pandas as pd

# Hindi Stopwords
HINDI_STOPWORDS = set([
    "के", "का", "एक", "की", "है", "यह", "और", "से", "हैं", "को", "पर", "इस", "होता",
    "कि", "जो", "कर", "मे", "गया", "करने", "किया", "लिये", "अपने", "ने", "बनी", "नहीं",
    "तो", "ही", "या", "हो", "था", "थे", "थी", "हुआ", "हुए", "हुई", "जब", "तब", "अब",
    "तक", "साथ", "सब", "कुछ", "भी", "दो", "गए", "गई", "वाले", "रहा", "रहे", "रही"
])

# Bengali Stopwords
BENGALI_STOPWORDS = set([
    "এই", "একটি", "এবং", "ও", "এর", "কে", "যে", "হয়", "হয়নি", "থেকে", "করতে",
    "করে", "করা", "হয়েছে", "আছে", "ছিল", "থাকবে", "বা", "না", "নয়", "তাহলে", "যদি",
    "তবে", "সব", "কিছু", "অনেক", "পর", "আগে", "সাথে", "দ্বারা", "উপর", "নিচে",
    "মধ্যে", "তিনি", "তারা", "আমরা", "তোমরা", "আমি", "তুমি", "সে", "কী", "কেন"
])

class IndicPreprocessor:
    def __init__(self, remove_stopwords=False):
        self.remove_stopwords = remove_stopwords

    def normalize_unicode(self, text: str) -> str:
        """Applies Unicode NFC normalization."""
        if not isinstance(text, str):
            return ""
        return unicodedata.normalize("NFC", text)

    def clean_social_text(self, text: str) -> str:
        """Removes URLs, mentions, repeated punctuation, and extra whitespace."""
        text = self.normalize_unicode(text)
        
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        
        # Remove user mentions (@username)
        text = re.sub(r"@\w+", "", text)
        
        # Remove hashtags symbol (#word -> word)
        text = re.sub(r"#", "", text)
        
        # Clean repetitive punctuation (e.g., "!!!!" -> "!")
        text = re.sub(r"([!?.।])\1+", r"\1", text)
        
        # Keep Indic characters (Devanagari, Bengali), Latin, numbers, and basic punctuation
        # Devanagari: \u0900-\u097F, Bengali: \u0980-\u09FF
        text = re.sub(r"[^\u0900-\u097F\u0980-\u09FFa-zA-Z0-9\s!?.,।]", " ", text)
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> list:
        """Tokenize Indic text by whitespace and punctuation boundaries."""
        cleaned = self.clean_social_text(text)
        tokens = cleaned.split()
        return tokens

    def preprocess(self, text: str, language: str = "Hindi") -> str:
        """Full pipeline: cleans, tokenizes, optionally filters stopwords."""
        tokens = self.tokenize(text)
        
        if self.remove_stopwords:
            stopwords = HINDI_STOPWORDS if language == "Hindi" else BENGALI_STOPWORDS
            tokens = [t for t in tokens if t not in stopwords]
            
        return " ".join(tokens)

def preprocess_splits(input_dir="d:/nlp/data", output_dir="d:/nlp/data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    preprocessor = IndicPreprocessor(remove_stopwords=False)
    
    for split in ["train.csv", "val.csv", "test.csv"]:
        path = os.path.join(input_dir, split)
        if not os.path.exists(path):
            continue
            
        # keep_default_na=False prevents 'None' from being parsed as NaN
        df = pd.read_csv(path, keep_default_na=False)
        print(f"Preprocessing {split} ({len(df)} samples)...")
        df["cleaned_text"] = df.apply(
            lambda row: preprocessor.preprocess(row["text"], row["language"]), axis=1
        )
        
        out_path = os.path.join(output_dir, split)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Saved preprocessed data to {out_path}")

if __name__ == "__main__":
    preprocess_splits()
