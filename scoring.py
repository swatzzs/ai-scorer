# scoring.py

import re
from collections import Counter

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# sentence-transformers imports
try:
    from sentence_transformers import SentenceTransformer, util
except Exception as e:
    print("Warning: sentence-transformers could not be imported:", e)
    SentenceTransformer = None
    util = None

# Try to import and initialize LanguageTool, but don't crash if it fails
try:
    import language_tool_python
    try:
        LANG_TOOL = language_tool_python.LanguageTool("en-US")
    except Exception as e:
        print("Warning: LanguageTool could not be initialized:", e)
        LANG_TOOL = None
except Exception as e:
    print("Warning: language_tool_python could not be imported:", e)
    language_tool_python = None
    LANG_TOOL = None

# -----------------------------
# Models (NLP)
# -----------------------------

# Sentence embedding model (for semantic similarity)
if SentenceTransformer is not None:
    try:
        EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print("Warning: SentenceTransformer model could not be loaded:", e)
        EMBED_MODEL = None
else:
    EMBED_MODEL = None

SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()


# -----------------------------
# Basic text helpers
# -----------------------------

def tokenize(text: str):
    # Simple word tokenizer: words only, lowercased
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    return tokens


def count_words(text: str) -> int:
    return len(tokenize(text))


# -----------------------------
# Salutation score (Content & Structure)
# Rubric from Excel:
#   No salutation -> 0
#   "Hi", "Hello" -> 2
#   "Good morning/afternoon/evening/day/Hello everyone" -> 4
#   Includes "I am excited to introduce / feeling great" -> 5
# Weight: 5
# -----------------------------

def salutation_score(text: str):
    t = text.lower().strip()
    score = 0
    level = "No salutation"

    # check first ~150 characters for salutation
    start = t[:150]

    if "excited to introduce" in start or "feeling great" in start:
        score = 5
        level = "Excellent"
    elif any(phrase in start for phrase in [
        "good morning",
        "good afternoon",
        "good evening",
        "good day",
        "hello everyone"
    ]):
        score = 4
        level = "Good"
    elif any(word in start.split() for word in ["hi", "hello"]):
        score = 2
        level = "Normal"

    return {
        "score": score,
        "max_score": 5,
        "level": level
    }


# -----------------------------
# Keyword presence (Content & Structure)
# Must-have (each 4 pts, total 20):
#   Name, Age, School/Class, Family, Hobbies/Interests
# Good-to-have (each 2 pts, total 10):
#   Origin Location, About Family, Aspirations, Fun fact/unique point, Strengths/Achievements
# Weight: 30 (20 + 10)
# -----------------------------

def keyword_presence_scores(text: str):
    t = text.lower()

    def contains_regex(pattern):
        return re.search(pattern, t) is not None

    must_keywords = {
        "name": lambda: ("name" in t or "myself" in t or "i am " in t),
        "age": lambda: contains_regex(r"\b\d{1,2}\s*(years old|year old|yrs old|yrs)\b"),
        "school/class": lambda: any(
            p in t for p in [
                "class ", "standard", "grade", "studying in", "school", "college", "university"
            ]
        ),
        "family": lambda: any(p in t for p in [
            "family", "father", "mother", "parents", "brother", "sister", "siblings"
        ]),
        "hobbies/interests": lambda: any(p in t for p in [
            "hobby", "hobbies", "interest", "interests", "in my free time",
            "i like to", "i love to", "i enjoy"
        ]),
    }

    good_keywords = {
        "origin/location": lambda: any(p in t for p in [
            "i am from", "i'm from", "i belong to", "my hometown", "from "
        ]),
        "about family (extra)": lambda: "family" in t and any(
            p in t for p in ["supportive", "loving", "small family", "joint family"]
        ),
        "aspirations/goals": lambda: any(p in t for p in [
            "i want to become", "my goal is", "i aspire", "i want to be", "my dream is"
        ]),
        "fun fact/unique point": lambda: any(p in t for p in [
            "fun fact", "something unique about me", "interesting fact"
        ]),
        "strengths/achievements": lambda: any(p in t for p in [
            "my strength", "my strengths", "my achievement", "i have achieved", "i achieved"
        ]),
    }

    must_found = []
    good_found = []

    must_score = 0
    for name, check in must_keywords.items():
        if check():
            must_found.append(name)
            must_score += 4

    good_score = 0
    for name, check in good_keywords.items():
        if check():
            good_found.append(name)
            good_score += 2

    must_score = min(must_score, 20)
    good_score = min(good_score, 10)

    total_score = must_score + good_score

    return {
        "score": total_score,
        "max_score": 30,
        "must_score": must_score,
        "good_score": good_score,
        "must_found": must_found,
        "good_found": good_found
    }


# -----------------------------
# Flow score (Content & Structure)
# Flow (order followed):
#   Salutation -> Basic details -> Additional details -> Closing
# Score: 5 if order followed, else 0
# Weight: 5
# -----------------------------

def flow_score(text: str):
    t = text.lower()

    def index_of_any(phrases):
        idxs = [t.find(p) for p in phrases if p in t]
        idxs = [i for i in idxs if i >= 0]
        return min(idxs) if idxs else None

    sal_idx = index_of_any(["good morning", "good afternoon", "good evening",
                            "hello", "hi", "hello everyone"])
    basic_idx = index_of_any(["my name is", "i am ", "i'm ", "myself "])
    add_idx = index_of_any(["in my free time", "my hobbies", "my hobby", "i like to", "i love to"])
    closing_idx = index_of_any(["thank you", "that's all", "nice to meet you"])

    indices = [("salutation", sal_idx), ("basic", basic_idx),
               ("additional", add_idx), ("closing", closing_idx)]

    present = [name for name, idx in indices if idx is not None]

    order_followed = False
    if len([idx for _, idx in indices if idx is not None]) >= 2:
        vals = [idx for _, idx in indices if idx is not None]
        order_followed = vals == sorted(vals)

    score = 5 if order_followed else 0

    return {
        "score": score,
        "max_score": 5,
        "present_sections": present,
        "order_followed": order_followed
    }


# -----------------------------
# Speech Rate score (Speech Rate)
# Rubric:
#   >161 WPM -> 2
#   141-160 -> 6
#   111-140 -> 10 (ideal)
#   81-110  -> 6
#   <80     -> 2
# Weight: 10
# For this case, we assume intro ≈ 1 minute if duration not provided.
# -----------------------------

def speech_rate_score(word_count: int, duration_seconds: float = 60.0):
    if duration_seconds <= 0:
        duration_seconds = 60.0
    wpm = word_count / (duration_seconds / 60.0)

    if wpm > 161:
        score = 2
        label = "Too fast"
    elif 141 <= wpm <= 160:
        score = 6
        label = "Fast"
    elif 111 <= wpm <= 140:
        score = 10
        label = "Ideal"
    elif 81 <= wpm <= 110:
        score = 6
        label = "Slow"
    else:
        score = 2
        label = "Too slow"

    return {
        "score": score,
        "max_score": 10,
        "wpm": round(wpm, 2),
        "label": label
    }


# -----------------------------
# Vocabulary richness (Language & Grammar)
# Uses TTR (type-token ratio)
# Rubric:
#   0.9–1.0  -> 10
#   0.7–0.89 -> 8
#   0.5–0.69 -> 6
#   0.3–0.49 -> 4
#   0–0.29   -> 2
# Weight: 10
# -----------------------------

def vocab_richness_score(text: str):
    tokens = tokenize(text)
    total = len(tokens)
    if total == 0:
        return {"score": 0, "max_score": 10, "ttr": 0.0}

    distinct = len(set(tokens))
    ttr = distinct / total

    if ttr >= 0.9:
        score = 10
    elif ttr >= 0.7:
        score = 8
    elif ttr >= 0.5:
        score = 6
    elif ttr >= 0.3:
        score = 4
    else:
        score = 2

    return {
        "score": score,
        "max_score": 10,
        "ttr": round(ttr, 3),
        "distinct_words": distinct,
        "total_words": total
    }


# -----------------------------
# Grammar score (Language & Grammar)
# Using LanguageTool when available
# Formula (from rubric):
#   GrammarScore = 1 - min(errors_per_100_words / 10, 1)
# Then mapped:
#   >0.9       -> 10
#   0.7–0.89   -> 8
#   0.5–0.69   -> 6
#   0.3–0.49   -> 4
#   <0.3       -> 2
# Weight: 10
# -----------------------------

def grammar_score(text: str):
    tokens = tokenize(text)
    total_words = len(tokens)

    # If no words, no score
    if total_words == 0:
        return {
            "score": 0,
            "max_score": 10,
            "errors": 0,
            "errors_per_100_words": 0.0,
            "raw_score": 0.0,
            "note": "Empty text, no grammar evaluation."
        }

    # If LanguageTool is not available, use a fallback neutral score
    if LANG_TOOL is None:
        return {
            "score": 6,  # neutral / moderate grammar
            "max_score": 10,
            "errors": 0,
            "errors_per_100_words": 0.0,
            "raw_score": 0.6,
            "note": "LanguageTool not available; using fallback grammar score."
        }

    # Normal LanguageTool-based scoring
    matches = LANG_TOOL.check(text)
    errors = len(matches)
    errors_per_100 = errors / total_words * 100

    raw = 1 - min(errors_per_100 / 10.0, 1.0)  # 10 errors per 100 -> 0

    if raw > 0.9:
        score = 10
    elif raw >= 0.7:
        score = 8
    elif raw >= 0.5:
        score = 6
    elif raw >= 0.3:
        score = 4
    else:
        score = 2

    return {
        "score": score,
        "max_score": 10,
        "errors": errors,
        "errors_per_100_words": round(errors_per_100, 2),
        "raw_score": round(raw, 3)
    }


# -----------------------------
# Clarity: Filler word rate
# Rubric:
#   Filler Word Rate = (filler words / total words) * 100
#   0–3          -> 15
#   4–6          -> 12
#   7–9          -> 9
#   10–12        -> 6
#   13 and above -> 3
# Weight: 15
# -----------------------------

FILLER_WORDS = {
    "um", "uh", "like", "you know", "kind of", "sort of",
    "actually", "basically", "literally"
}


def clarity_filler_score(text: str):
    tokens = tokenize(text)
    total = len(tokens)
    if total == 0:
        return {
            "score": 0,
            "max_score": 15,
            "filler_count": 0,
            "filler_rate": 0.0
        }

    t = text.lower()
    filler_count = 0
    # count phrase fillers
    for phrase in ["you know", "kind of", "sort of"]:
        filler_count += t.count(phrase)

    # word fillers
    for token in tokens:
        if token in {"um", "uh", "like", "actually", "basically", "literally"}:
            filler_count += 1

    filler_rate = filler_count / total * 100

    if filler_rate <= 3:
        score = 15
        label = "Excellent"
    elif filler_rate <= 6:
        score = 12
        label = "Good"
    elif filler_rate <= 9:
        score = 9
        label = "Average"
    elif filler_rate <= 12:
        score = 6
        label = "Below average"
    else:
        score = 3
        label = "Needs improvement"

    return {
        "score": score,
        "max_score": 15,
        "filler_count": filler_count,
        "filler_rate": round(filler_rate, 2),
        "label": label
    }


# -----------------------------
# Engagement: Sentiment / positivity
# Using VADER positive score
# Rubric:
#   >=0.9      -> 15
#   0.7–0.89   -> 12
#   0.5–0.69   -> 9
#   0.3–0.49   -> 6
#   <0.3       -> 3
# Weight: 15
# -----------------------------

def engagement_sentiment_score(text: str):
    scores = SENTIMENT_ANALYZER.polarity_scores(text)
    pos = scores["pos"]

    if pos >= 0.9:
        score = 15
    elif pos >= 0.7:
        score = 12
    elif pos >= 0.5:
        score = 9
    elif pos >= 0.3:
        score = 6
    else:
        score = 3

    return {
        "score": score,
        "max_score": 15,
        "positive_prob": round(pos, 3),
        "vader_scores": scores
    }


# -----------------------------
# Semantic similarity (NLP-based)
# For each criterion, compare transcript to a short description prompt.
# -----------------------------

CRITERION_DESCRIPTIONS = {
    "salutation": "The introduction starts with a warm, appropriate greeting such as good morning or hello everyone, ideally with enthusiasm.",
    "keywords": "A self introduction that clearly mentions the speaker's name, age, class or school, family, hobbies, goals and some unique personal details.",
    "flow": "The introduction follows a logical order: salutation, basic details like name and class, additional details such as hobbies and goals, and a closing line.",
    "speech_rate": "The speaker talks at a natural, comfortable speed, not too fast or too slow, around 110 to 140 words per minute.",
    "vocab": "The speaker uses a rich and varied vocabulary instead of repeating the same simple words.",
    "grammar": "The introduction has very few grammar mistakes, good sentence structure and correct usage of tenses and articles.",
    "clarity": "The speech is clear and confident with very few filler words like um, uh or like.",
    "engagement": "The tone feels positive, enthusiastic and engaging, making the listener interested in the speaker."
}


def semantic_similarity(text: str, key: str):
    desc = CRITERION_DESCRIPTIONS.get(key)
    if not desc:
        return None

    # If embedding model is not available, return neutral similarity
    if EMBED_MODEL is None or util is None:
        return 0.5

    embeddings = EMBED_MODEL.encode([text, desc], convert_to_tensor=True)
    sim = float(util.cos_sim(embeddings[0], embeddings[1]))
    # cosine similarity is often in -1..1; clamp to 0..1
    sim_norm = max(0.0, min(1.0, (sim + 1) / 2.0))
    return round(sim_norm, 3)


# -----------------------------
# Main scoring orchestrator
# Combines:
#   - Rule-based metrics -> raw rubric score
#   - Semantic similarity -> used as secondary factor (30%)
#
# For each criterion:
#   final_metric_score = weight * (0.7 * rule_fraction + 0.3 * similarity)
#
# Overall score = sum of all criterion final scores (approx 0–100)
# -----------------------------

def score_transcript(text: str, duration_seconds: float | None = None):
    if duration_seconds is None:
        duration_seconds = 60.0  # assumption for 1-min intro

    total_words = count_words(text)

    # Compute rule-based scores
    sal = salutation_score(text)
    kw = keyword_presence_scores(text)
    flow = flow_score(text)
    speech = speech_rate_score(total_words, duration_seconds)
    vocab = vocab_richness_score(text)
    gram = grammar_score(text)
    clarity = clarity_filler_score(text)
    engage = engagement_sentiment_score(text)

    # Semantic similarities (fallback 0.5 if None)
    sal_sim = semantic_similarity(text, "salutation") or 0.5
    kw_sim = semantic_similarity(text, "keywords") or 0.5
    flow_sim = semantic_similarity(text, "flow") or 0.5
    speech_sim = semantic_similarity(text, "speech_rate") or 0.5
    vocab_sim = semantic_similarity(text, "vocab") or 0.5
    gram_sim = semantic_similarity(text, "grammar") or 0.5
    clarity_sim = semantic_similarity(text, "clarity") or 0.5
    engage_sim = semantic_similarity(text, "engagement") or 0.5

    def combine(rule_obj, sim):
        rule_fraction = rule_obj["score"] / rule_obj["max_score"] if rule_obj["max_score"] > 0 else 0
        combined = rule_obj["max_score"] * (0.7 * rule_fraction + 0.3 * sim)
        return round(combined, 2)

    criteria = []

    def add_criterion(name, group, key, rule_obj, sim, extra_feedback=""):
        final_score = combine(rule_obj, sim)
        feedback_bits = []

        # generic feedback
        if rule_obj["max_score"] > 0:
            pct = rule_obj["score"] / rule_obj["max_score"]
        else:
            pct = 0

        if pct >= 0.8:
            feedback_bits.append("This aspect is strong overall.")
        elif pct >= 0.5:
            feedback_bits.append("This aspect is okay, but there is room for improvement.")
        else:
            feedback_bits.append("This aspect needs significant improvement.")

        feedback_bits.append(f"Semantic similarity to rubric: {sim}.")

        if extra_feedback:
            feedback_bits.append(extra_feedback)

        criteria.append({
            "name": name,
            "group": group,
            "rule_score": rule_obj["score"],
            "rule_max_score": rule_obj["max_score"],
            "final_score": final_score,
            "semantic_similarity": sim,
            "details": rule_obj,
            "feedback": " ".join(feedback_bits)
        })

    # Content & Structure
    add_criterion(
        "Salutation Level",
        "Content & Structure",
        "salutation",
        sal,
        sal_sim,
        extra_feedback=f"Detected level: {sal['level']}."
    )
    add_criterion(
        "Keyword Presence",
        "Content & Structure",
        "keywords",
        {"score": kw["score"], "max_score": kw["max_score"]},
        kw_sim,
        extra_feedback=f"Must-haves found: {kw['must_found']}. Good-to-haves found: {kw['good_found']}."
    )
    add_criterion(
        "Flow",
        "Content & Structure",
        "flow",
        flow,
        flow_sim,
        extra_feedback=f"Sections present: {flow['present_sections']}. Order followed: {flow['order_followed']}."
    )

    # Speech Rate
    add_criterion(
        "Speech Rate (WPM)",
        "Speech Rate",
        "speech_rate",
        speech,
        speech_sim,
        extra_feedback=f"Estimated WPM: {speech['wpm']} ({speech['label']})."
    )

    # Language & Grammar
    add_criterion(
        "Vocabulary Richness (TTR)",
        "Language & Grammar",
        "vocab",
        vocab,
        vocab_sim,
        extra_feedback=f"TTR: {vocab['ttr']} with {vocab['distinct_words']} distinct words."
    )
    add_criterion(
        "Grammar Accuracy",
        "Language & Grammar",
        "grammar",
        gram,
        gram_sim,
        extra_feedback=f"Grammar errors: {gram['errors']} (≈ {gram.get('errors_per_100_words', 0)} per 100 words)."
    )

    # Clarity
    add_criterion(
        "Clarity (Filler Words)",
        "Clarity",
        "clarity",
        clarity,
        clarity_sim,
        extra_feedback=f"Filler words: {clarity['filler_count']} (≈ {clarity['filler_rate']}% of words, {clarity.get('label', '')})."
    )

    # Engagement
    add_criterion(
        "Engagement / Positivity",
        "Engagement",
        "engagement",
        engage,
        engage_sim,
        extra_feedback=f"Positive sentiment score: {engage['positive_prob']}."
    )

    overall_score = round(sum(c["final_score"] for c in criteria), 2)
    # clamp 0–100
    overall_score = max(0.0, min(100.0, overall_score))

    return {
        "overall_score": overall_score,
        "words": total_words,
        "criteria": criteria
    }
