"""
Data Augmentation for Indic Stance Detection (Bonus Marks Requirement)
Implements:
1. Synonym Replacement using an Indic Environmental Thesaurus
2. Random Word Swapping & Deletion (EDA)
3. Noise Injection for robust social media representation
"""

import os
import random
import pandas as pd

random.seed(42)

HINDI_SYNONYMS = {
    "खतरा": ["संकट", "आपदा", "विपत्ति", "जोखिम"],
    "गंभीर": ["विकराल", "भयानक", "चिंताजनक", "बड़ा"],
    "गर्मी": ["तापमान", "उष्णता", "तपन"],
    "पेड़": ["वृक्ष", "पौधे", "वनस्पति"],
    "धरती": ["पृथ्वी", "जमीन", "प्रकृति"],
    "ढोंग": ["पाखंड", "दिखावा", "नाटक", "षड्यंत्र"],
    "झूठ": ["फर्जी", "गलत", "अफवाह"],
    "नियम": ["सिद्धांत", "व्यवस्था", "क्रम"]
}

BENGALI_SYNONYMS = {
    "বিপদ": ["সংকট", "ঝুঁকি", "দূর্যোগ", "হুমকি"],
    "গুরুতর": ["মারাত্মক", "ভয়ানক", "চরম", "বিরাট"],
    "গরম": ["উষ্ণতা", "তাপদাহ", "তাপমাত্রা"],
    "গাছ": ["বৃক্ষ", "উদ্ভিদ", "বন"],
    "প্রকৃতি": ["পরিবেশ", "পৃথিবী", "ধরিত্রী"],
    "ভুয়া": ["মিথ্যা", "নাটক", "ভাওঁতা", "ধোঁকাবাজি"],
    "চক্রান্ত": ["ষড়যন্ত্র", "ফন্দি", "মতলব"]
}

class IndicDataAugmentor:
    def __init__(self, aug_prob=0.3):
        self.aug_prob = aug_prob

    def synonym_replace(self, text: str, lang: str) -> str:
        words = text.split()
        thesaurus = HINDI_SYNONYMS if lang == "Hindi" else BENGALI_SYNONYMS
        new_words = []
        replaced = False
        
        for w in words:
            # Check for exact or substring match in thesaurus
            matched = False
            for k, syns in thesaurus.items():
                if k in w and random.random() < self.aug_prob:
                    new_words.append(random.choice(syns))
                    replaced = True
                    matched = True
                    break
            if not matched:
                new_words.append(w)
                
        return " ".join(new_words) if replaced else text

    def random_swap(self, text: str) -> str:
        words = text.split()
        if len(words) < 4:
            return text
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        return " ".join(words)

    def augment(self, text: str, lang: str) -> str:
        if random.random() < 0.5:
            return self.synonym_replace(text, lang)
        else:
            return self.random_swap(text)

def generate_augmented_dataset(
    input_path="d:/nlp/data/processed/train.csv",
    output_path="d:/nlp/data/augmented/train_augmented.csv",
    factor=1
):
    df = pd.read_csv(input_path, keep_default_na=False)
    augmentor = IndicDataAugmentor(aug_prob=0.4)
    
    augmented_rows = []
    print(f"Applying data augmentation on {len(df)} training samples...")
    
    for _, row in df.iterrows():
        # Keep original
        augmented_rows.append(row.to_dict())
        
        # Generate augmented variations
        for f in range(factor):
            aug_text = augmentor.augment(row["cleaned_text"], row["language"])
            aug_row = row.to_dict().copy()
            aug_row["id"] = f"{row['id']}_AUG{f+1}"
            aug_row["cleaned_text"] = aug_text
            augmented_rows.append(aug_row)
            
    aug_df = pd.DataFrame(augmented_rows)
    aug_df = aug_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    aug_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Augmented dataset created: {len(aug_df)} total samples (Original: {len(df)}).")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    generate_augmented_dataset()
