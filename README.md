# Nirmaan AI – Spoken Communication Scoring Tool

This project is a case-study implementation for the Nirmaan AI Intern Communication Program.

It takes a **transcript** of a student's self-introduction (speech) as input and outputs:

- An **overall score** (0–100)
- **Per-criterion scores** based on a rubric
- **Per-criterion feedback**, including:
  - Rule-based score
  - NLP-based semantic similarity score
  - Suggestions on how to improve

The rubric is derived from the provided Excel: `Case study for interns.xlsx`.

---

## 1. Features

1. **Input**
   - Paste transcript into a text area (web UI).
   - Optionally provide approximate audio duration in seconds.

2. **Rule-based scoring**
   - Salutation type (Hi/Hello vs more formal/enthusiastic).
   - Presence of required/optional keywords (name, age, school, family, hobbies, goals, etc.).
   - Flow/order of introduction (Salutation → Basic details → Additional details → Closing).
   - Speech rate (words per minute, approximated with duration).
   - Vocabulary richness using **type-token ratio (TTR)**.
   - Grammar errors using **LanguageTool**.
   - Filler word rate (e.g., "um", "uh", "like", "you know").
   - Sentiment/positivity using **VADER**.

3. **NLP-based semantic scoring**
   - Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to compute semantic similarity between:
     - The transcript
     - A short natural-language description of each rubric criterion  
   - Similarity is in the range ~0–1.

4. **Rubric-driven weighting**
   - We follow the Excel rubric structure:
     - Content & Structure – 40 points
       - Salutation Level (5)
       - Keyword Presence (30)
       - Flow (5)
     - Speech Rate – 10 points
     - Language & Grammar – 20 points
       - Vocabulary Richness (10)
       - Grammar Accuracy (10)
     - Clarity – 15 points
       - Filler Word Rate (15)
     - Engagement – 15 points
       - Sentiment / Positivity (15)
   - Total possible rule-based: 100.

---

## 2. Scoring Formula

For each metric, we first compute a **rule-based score** using the rubric, and then combine it with semantic similarity.

Let:

- `R` = rule-based score for the metric
- `R_max` = maximum possible rule-based score for that metric (from rubric)
- `sim` = semantic similarity between transcript and the rubric description of that metric (0–1)

We compute:

```text
rule_fraction = R / R_max
final_metric_score = R_max * (0.7 * rule_fraction + 0.3 * sim)
