"""
Indic Stance Detection - Preprocessing & Normalization Pipeline
Step 3 in the NLP Project Pipeline

Designed for Hindi (Devanagari) & Bengali (Bangla) social media texts.
Target Claim: "Climate change and global warming is a serious concern."

Pipeline Steps:
1. Unicode NFC Normalization: Canonical representation of Indic characters & conjuncts.
2. Social Artifact Cleaning: Strips URLs, mentions, HTML tags; preserves hashtags as text.
3. Character De-elongation: Collapses 3+ repeated characters down to 2 ("नहींइइइइ" -> "नहींइइ").
   Preserves valid Indic double consonants (e.g. "पक्का", "দিল্লি") untouched.
4. Emoji Preservation & Normalization:
   - Preserves emojis as high-signal stance tokens (🌱, 🌍, 🙄, 😂, 🤡, ❤️, 🤦‍♂️).
   - Emoji Spacing: Guarantees whitespace separation so emojis are not glued to Indic words.
   - Emoji Squashing: Collapses 3+ identical emojis to max 2 to prevent feature explosion.
   - Full grapheme cluster support (handles variation selectors \uFE0F and ZWJ sequences).
5. Punctuation Normalization: Collapses excessive repeated punctuation ("!!!!????" -> "!?").
6. Indic Tokenizer: Extracts Indic words, emojis, numbers, and key punctuation.
"""

import sys
import os
import re
import html
import unicodedata
import pandas as pd

# Fix Windows console UTF-8 output encoding
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# --------------------------------------------------------------------------
# REGEX PATTERNS
# --------------------------------------------------------------------------

# URLs & Social Handles
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+", re.UNICODE)
HASHTAG_PATTERN = re.compile(r"#([\w\u0900-\u097F\u0980-\u09FF]+)", re.UNICODE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

# Base Emoji Characters
EMOJI_BASE = (
    r"[\U0001F1E6-\U0001F1FF]"      # Flags
    r"|[\U0001F300-\U0001F5FF]"     # Symbols & pictographs
    r"|[\U0001F600-\U0001F64F]"     # Emoticons (smileys, clowns, etc.)
    r"|[\U0001F680-\U0001F6FF]"     # Transport & map
    r"|[\U0001F700-\U0001F77F]"     # Alchemical symbols
    r"|[\U0001F780-\U0001F7FF]"     # Geometric Shapes Extended
    r"|[\U0001F800-\U0001F8FF]"     # Supplemental Arrows-C
    r"|[\U0001F900-\U0001F9FF]"     # Supplemental Symbols & Pictographs (facepalms, skulls)
    r"|[\U0001FA00-\U0001FA6F]"     # Chess Symbols
    r"|[\U0001FA70-\U0001FAFF]"     # Symbols & Pictographs Extended-A
    r"|[\u2600-\u26FF]"             # Misc symbols (e.g. ☀️, ⚠️)
    r"|[\u2700-\u27BF]"             # Dingbats (e.g. ❄️, ✈️, ❤️)
    r"|[\u2300-\u23FF]"             # Misc technical
    r"|[\u2B50]"                    # Star
)

# Comprehensive Emoji Sequence (handles Variation Selectors \uFE0E/\uFE0F, Skin tones, and ZWJ joins)
EMOJI_SEQUENCE = re.compile(
    rf"(?:(?:{EMOJI_BASE})[\uFE0E\uFE0F]?(?:[\U0001F3FB-\U0001F3FF])?(?:\u200D(?:(?:{EMOJI_BASE})[\uFE0E\uFE0F]?(?:[\U0001F3FB-\U0001F3FF])?))*)",
    re.UNICODE
)

# Character elongation pattern specifically targeting Indic letters and Latin word characters:
# Hindi: \u0900-\u097F, Bengali: \u0980-\u09FF, Latin: a-zA-Z
ELONGATION_PATTERN = re.compile(
    r"([\u0900-\u097F\u0980-\u09FFa-zA-Z])\1{2,}",
    re.UNICODE
)

# Punctuation normalization patterns
PUNC_REPEAT_EXCLAM = re.compile(r"!{2,}")
PUNC_REPEAT_QUESTION = re.compile(r"\?{2,}")
PUNC_REPEAT_DANDA = re.compile(r"[।]{2,}")
PUNC_REPEAT_DOTS = re.compile(r"\.{3,}")
PUNC_MIXED_EXCLAM_QUEST = re.compile(r"!\?+|\?!+")

# Invisible control and zero-width characters (excluding ZWJ \u200D and ZWNJ \u200C which belong to Indic conjuncts and emojis)
INVISIBLE_CHARS_PATTERN = re.compile(r"[\u200b\u200e\u200f\ufeff\u00a0]")

# Mobile keyboard ASCII Pipe to Indic Danda
KEYBOARD_PIPE_PATTERN = re.compile(r"\s*\|\s*")

# Typography normalization map (smart quotes to standard, em/en dashes to hyphen)
TYPOGRAPHY_MAP = {
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "—": " - ", "–": " - "
}

# Indic Word + Emoji Tokenizer Pattern
# Captures Devanagari words, Bengali words, Latin words, digits, full emoji sequences, and punctuation
INDIC_TOKEN_PATTERN = re.compile(
    rf"(?:(?:{EMOJI_BASE})[\uFE0E\uFE0F]?(?:[\U0001F3FB-\U0001F3FF])?(?:\u200D(?:(?:{EMOJI_BASE})[\uFE0E\uFE0F]?(?:[\U0001F3FB-\U0001F3FF])?))*)"
    r"|[\u0900-\u097F]+"               # Hindi / Devanagari words
    r"|[\u0980-\u09FF]+"               # Bengali words
    r"|[a-zA-Z]+"                      # English / Latin words
    r"|\d+"                            # Numbers
    r"|[!?।,;:\"'()\-]+",              # Punctuation tokens
    re.UNICODE
)


# --------------------------------------------------------------------------
# PIPELINE FUNCTIONS
# --------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """Canonical Unicode NFC normalization for Indic scripts."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text)


def clean_social_artifacts(text: str) -> str:
    """Removes URLs, mentions, HTML tags, and cleans hashtags into plain text."""
    # Decode HTML entities (&amp; -> &, &lt; -> <, etc.)
    text = html.unescape(text)
    # Strip HTML tags if any
    text = HTML_TAG_PATTERN.sub(" ", text)
    # Strip URLs
    text = URL_PATTERN.sub(" ", text)
    # Strip @mentions
    text = MENTION_PATTERN.sub(" ", text)
    # Keep hashtag body as text (e.g. #ग्लोबलवार्मिंग -> ग्लोबलवार्मिंग)
    text = HASHTAG_PATTERN.sub(r" \1 ", text)
    return text


def squash_character_elongation(text: str) -> str:
    """
    Collapses 3 or more consecutive identical characters down to 2.
    Example: 'नहींइइइइ' -> 'नहींइइ', 'সত্যিইইই' -> 'সত্যিইই'.
    Preserves valid Indic double consonants like 'पक्का' or 'দিল্লি' untouched.
    """
    return ELONGATION_PATTERN.sub(r"\1\1", text)


def normalize_emojis(text: str) -> str:
    """
    1. Spacing: Ensures whitespace around emoji sequences so tokenizers treat them independently.
    2. Squashing: Collapses runs of 3+ identical emojis down to 2 (e.g. '😂😂😂😂' -> '😂 😂').
       Maintains full emoji grapheme clusters (including variation selectors and ZWJ joins).
    """
    # Step A: Ensure space around all emoji sequences
    spaced = EMOJI_SEQUENCE.sub(r" \g<0> ", text)
    
    # Step B: Squash 3+ consecutive identical emojis down to 2
    tokens = [tok.strip() for tok in spaced.split() if tok.strip()]
    squashed_toks = []
    prev_tok = None
    rep_count = 0
    
    for tok in tokens:
        if tok == prev_tok and EMOJI_SEQUENCE.fullmatch(tok):
            rep_count += 1
            if rep_count <= 2:
                squashed_toks.append(tok)
        else:
            prev_tok = tok
            rep_count = 1
            squashed_toks.append(tok)
            
    return " ".join(squashed_toks)


def normalize_punctuation(text: str) -> str:
    """
    Normalizes excessive punctuation without losing emotive intent.
    - Multiple ! -> !!
    - Multiple ? -> ??
    - Multiple । (Danda) -> ।।
    - Multiple . -> ...
    - Mixed !? / ?! -> !?
    """
    text = PUNC_REPEAT_EXCLAM.sub("!!", text)
    text = PUNC_REPEAT_QUESTION.sub("??", text)
    text = PUNC_REPEAT_DANDA.sub("।।", text)
    text = PUNC_REPEAT_DOTS.sub("...", text)
    text = PUNC_MIXED_EXCLAM_QUEST.sub("!?", text)
    return text


def clean_indic_text(text: str, preserve_emojis: bool = True) -> str:
    """
    Master preprocessing pipeline for Indic text:
    1. Unicode NFC normalization
    2. Invisible zero-width character sanitization (\u200B, \u00A0, \uFEFF)
    3. Typography standardization (curly quotes -> straight, em-dashes -> hyphen)
    4. Mobile keyboard pipe normalization (| -> Indic Danda ।)
    5. Social artifact cleaning (URLs, mentions, HTML)
    6. Character de-elongation (max 2 repeated chars, protecting double consonants)
    7. Emoji spacing and squashing (preserving full grapheme clusters)
    8. Punctuation normalization
    9. Safe Latin lowercasing (unifies acronyms like IPCC/CO2; leaves Indic untouched)
    10. Linguistic integrity: Chandrabindu (ँ, ঁ) is strictly preserved (distinct from Anusvara)
    11. Whitespace cleanup
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Unicode NFC normalization
    cleaned = normalize_unicode(text)
    
    # 2. Invisible character sanitization
    cleaned = INVISIBLE_CHARS_PATTERN.sub(" ", cleaned)
    
    # 3. Typography standardization
    for orig, repl in TYPOGRAPHY_MAP.items():
        cleaned = cleaned.replace(orig, repl)
        
    # 4. Keyboard pipe normalization (| -> Indic Danda ।)
    cleaned = KEYBOARD_PIPE_PATTERN.sub(" । ", cleaned)
    
    # 5. Social artifact cleaning
    cleaned = clean_social_artifacts(cleaned)
    
    # 6. De-elongation
    cleaned = squash_character_elongation(cleaned)
    
    # 7. Emoji handling
    if preserve_emojis:
        cleaned = normalize_emojis(cleaned)
    else:
        cleaned = EMOJI_SEQUENCE.sub(" ", cleaned)
        
    # 8. Punctuation normalization
    cleaned = normalize_punctuation(cleaned)
    
    # 9. Safe Latin lowercasing (applies only to Latin letters a-z; Indic scripts have no cases)
    cleaned = cleaned.lower()
    
    # 10. Whitespace cleanup
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    return cleaned


def indic_tokenize(text: str) -> list[str]:
    """
    Tokenizer tailored for Indic languages + emojis + punctuation.
    Splits text into individual words, full emoji sequences, numbers, and punctuation marks.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    return INDIC_TOKEN_PATTERN.findall(text)


# --------------------------------------------------------------------------
# DATASET BATCH PROCESSOR
# --------------------------------------------------------------------------

def process_and_save_datasets(data_dir: str = "d:/nlp/data", processed_dir: str = "d:/nlp/data/processed"):
    """
    Applies the preprocessing pipeline to all project datasets:
    - train.csv
    - val.csv
    - test.csv
    - hidden_test_unlabeled.csv
    - raw/indic_stance_corpus_3200.csv
    
    Outputs cleaned files with 'cleaned_text' column to data/processed/.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    tasks = [
        ("train.csv", True),
        ("val.csv", True),
        ("test.csv", True),
        ("hidden_test_unlabeled.csv", False),
    ]
    
    print("=" * 75)
    print("INDIC STANCE DETECTION - PREPROCESSING PIPELINE EXECUTION")
    print("=" * 75)
    
    stats_summary = []
    
    for filename, is_labeled in tasks:
        in_path = os.path.join(data_dir, filename)
        out_path = os.path.join(processed_dir, filename)
        
        if not os.path.exists(in_path):
            print(f"[WARNING] File not found: {in_path}")
            continue
            
        df = pd.read_csv(in_path, keep_default_na=False)
        print(f"\nProcessing {filename} ({len(df)} rows)...")
        
        # Apply preprocessing
        df["cleaned_text"] = df["text"].apply(clean_indic_text)
        
        # Calculate statistics
        orig_char_count = df["text"].str.len().sum()
        clean_char_count = df["cleaned_text"].str.len().sum()
        
        # Token count
        all_tokens = [tok for t in df["cleaned_text"] for tok in indic_tokenize(t)]
        vocab = set(all_tokens)
        emoji_tokens = [tok for tok in all_tokens if EMOJI_SEQUENCE.fullmatch(tok)]
        
        # Reorder columns: id, text, cleaned_text, language, [stance]
        if is_labeled:
            cols = ["id", "text", "cleaned_text", "language", "stance"]
        else:
            cols = ["id", "text", "cleaned_text", "language"]
            
        df = df[cols]
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        
        stats_summary.append({
            "File": filename,
            "Rows": len(df),
            "Total Tokens": len(all_tokens),
            "Unique Vocab": len(vocab),
            "Emoji Tokens": len(emoji_tokens),
            "Unique Emojis": len(set(emoji_tokens)),
            "Char Delta": f"{clean_char_count - orig_char_count:+d}"
        })
        print(f"  -> Saved to {out_path}")
        print(f"  -> Total Tokens: {len(all_tokens):,d} | Unique Vocab: {len(vocab):,d} | Emoji Tokens: {len(emoji_tokens):,d}")

    # Also process raw 3200 corpus for full analysis & research report
    raw_in = os.path.join(data_dir, "raw", "indic_stance_corpus_3200.csv")
    if os.path.exists(raw_in):
        df_raw = pd.read_csv(raw_in, keep_default_na=False)
        df_raw["cleaned_text"] = df_raw["text"].apply(clean_indic_text)
        raw_out = os.path.join(processed_dir, "indic_stance_corpus_3200_cleaned.csv")
        df_raw.to_csv(raw_out, index=False, encoding="utf-8-sig")
        print(f"\nSaved full processed corpus to {raw_out} ({len(df_raw)} rows)")

    # Print summary table
    print("\n" + "=" * 75)
    print("PREPROCESSING SUMMARY TABLE")
    print("=" * 75)
    summary_df = pd.DataFrame(stats_summary)
    print(summary_df.to_string(index=False))

    # Demonstrate sample transformations
    print("\n" + "=" * 75)
    print("SAMPLE TRANSFORMATIONS (RAW -> CLEANED)")
    print("=" * 75)
    
    sample_df = pd.read_csv(os.path.join(processed_dir, "train.csv"), keep_default_na=False)
    # Pick diverse samples: sarcasm, emojis, weather, elongation
    demo_indices = [0, 42, 105, 520, 890]
    for idx in demo_indices:
        if idx < len(sample_df):
            row = sample_df.iloc[idx]
            print(f"[{row['language']} - {row['stance']}]")
            print(f"  RAW:     {row['text']}")
            print(f"  CLEANED: {row['cleaned_text']}")
            tokens = indic_tokenize(row["cleaned_text"])
            print(f"  TOKENS:  {tokens[:8]}..." if len(tokens) > 8 else f"  TOKENS:  {tokens}")
            print("-" * 75)


if __name__ == "__main__":
    process_and_save_datasets()
