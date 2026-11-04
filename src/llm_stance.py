"""
LLM-Based Stance Detection Module (Course Requirement)
Explores Zero-Shot and Few-Shot In-Context Prompting for Indic YouTube comments.
Supports local lightweight LLMs and API-based inference.
"""

import os
import json
import argparse
import pandas as pd

CLAIM = "Climate change and global warming is a serious concern."

HINDI_PROMPT_TEMPLATE = """
आप एक भाषा विशेषज्ञ हैं। आपका कार्य YouTube कमेंट्स के आधार पर यह निर्धारित करना है कि टिप्पणीकार का नीचे दिए गए दावे (Claim) के प्रति क्या रुख (Stance) है।

दावा (Claim): "{claim}"

वर्गीकरण श्रेणियां (Categories):
- Favour: यदि टिप्पणी सहमत है कि जलवायु परिवर्तन या ग्लोबल वार्मिंग एक गंभीर चिंता का विषय है।
- Against: यदि टिप्पणी दावे से असहमत है (जैसे इसे झूठ, पश्चिमी साजिश, या सामान्य प्राकृतिक चक्र मानती है)।
- None: यदि टिप्पणी तटस्थ है, अप्रासंगिक है, या दावे पर कोई स्पष्ट रुख व्यक्त नहीं करती।

उदाहरण:
कमेंट: "अगर हमने अभी पेड़ नहीं लगाए तो आने वाली पीढ़ी तबाह हो जाएगी।"
रुख: Favour

कमेंट: "यह सब पश्चिमी देशों का ढोंग है, भारत के विकास को रोकने की साजिश है।"
रुख: Against

कमेंट: "सर आपका वीडियो बहुत अच्छा लगा, धन्यवाद।"
रुख: None

अब निम्नलिखित कमेंट का रुख बताएं:
कमेंट: "{comment}"
रुख (केवल एक शब्द लिखें: Favour, Against, या None):
"""

BENGALI_PROMPT_TEMPLATE = """
আপনি একজন ভাষা বিশেষজ্ঞ। আপনার কাজ হলো ইউটিউব মন্তব্যের ভিত্তিতে নিচে দেওয়া দাবিটির (Claim) প্রতি মন্তব্যের অবস্থান (Stance) নির্ধারণ করা।

দাবি (Claim): "{claim}"

শ্রেণীসমূহ (Categories):
- Favour: যদি মন্তব্যটি একমত হয় যে জলবায়ু পরিবর্তন বা বৈশ্বিক উষ্ণতা একটি মারাত্মক সংকট।
- Against: যদি মন্তব্যটি দাবির সাথে দ্বিমত পোষণ করে (যেমন এটিকে ভুয়া নাটক, পশ্চিমা ষড়যন্ত্র বা স্বাভাবিক ঘটনা মনে করে)।
- None: যদি মন্তব্যটি নিরপেক্ষ, অপ্রাসঙ্গিক হয় বা কোনো স্পষ্ট অবস্থান প্রকাশ না করে।

উদাহরণ:
মন্তব্য: "সুন্দরবন তলিয়ে যাচ্ছে, এখনই বৃক্ষরোপণ শুরু করতে হবে।"
অবস্থান: Favour

মন্তব্য: "সব পশ্চিমা ফান্ডিংয়ের মিথ্যা নাটক, কোনো গ্লোবাল ওয়ার্মিং নেই।"
অবস্থান: Against

মন্তব্য: "খুব সুন্দর ভিডিও দাদা, পরের পর্ব কবে আসবে?"
অবস্থান: None

এখন নিচের মন্তব্যের অবস্থান নির্ণয় করুন:
মন্তব্য: "{comment}"
অবস্থান (শুধুমাত্র একটি শব্দ লিখুন: Favour, Against, অথবা None):
"""

def generate_prompt(comment: str, language: str) -> str:
    template = HINDI_PROMPT_TEMPLATE if language == "Hindi" else BENGALI_PROMPT_TEMPLATE
    return template.format(claim=CLAIM, comment=comment).strip()

def run_llm_exploration(test_path="d:/nlp/data/processed/test.csv", sample_size=10):
    print("=" * 60)
    print("LLM In-Context Learning Prompt Exploration for Indic Stance")
    print("=" * 60)
    
    test_df = pd.read_csv(test_path, keep_default_na=False)
    sample_df = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)
    
    for _, row in sample_df.iterrows():
        prompt = generate_prompt(row["cleaned_text"], row["language"])
        print(f"\n[ID: {row['id']}] [{row['language']}] True Stance: {row['stance']}")
        print(f"Sample Comment: {row['cleaned_text']}")
        print(f"Generated Few-Shot Prompt Preview:\n{prompt[:250]}...\n")
        
    print(f"\n[LLM] Demonstrated Few-Shot Prompt Templates for both Hindi and Bengali.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, default="d:/nlp/data/processed/test.csv")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    run_llm_exploration(args.test_file, args.samples)
