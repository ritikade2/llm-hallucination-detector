import pandas as pd 
import re

# Load dataset
df = pd.read_csv("halueval_qa_samples.csv")

#--------------------------------------------------------------
# SIGNAL 1: Answer length and knowledge length ratio
# Hallucinated answers tend to be longer relative to source knowledge
#--------------------------------------------------------------
df['answer_len'] = df['answer'].str.split().str.len()
df['knowledge_len'] = df['knowledge'].str.split().str.len()
df['length_ratio'] = df['answer_len'] / df['knowledge_len']

#--------------------------------------------------------------
# SIGNAL 2: Unknown word rate
# Hallucinated answers introduce words not found in the source knowledge
#--------------------------------------------------------------
def unknown_word_rate(row):
    knowledge_words = set(str(row['knowledge']).lower().split())
    answer_words = set(str(row['answer']).lower().split())
    if not answer_words:
        return 0
    unknown = answer_words - knowledge_words
    return len(unknown) / len(answer_words)

df['unknown_word_rate'] = df.apply(unknown_word_rate, axis = 1)

#--------------------------------------------------------------
# SIGNAL 3: Question-answer word overlap
# Hallucinated answers repeat question words back - classic sign of 
# a model padding or fabricating rather than answering from source.
#--------------------------------------------------------------
def question_answer_overlap(row):
    question_words = set(str(row['question']).lower().split())
    answer_words = set(str(row['answer']).lower().split())
    if not question_words:
        return 0
    overlap = question_words & answer_words
    return len(overlap) / len(question_words)

df['qa_overlap'] = df.apply(question_answer_overlap, axis=1)

#--------------------------------------------------------------
# SIGNAL 4: Numeric inconsistency
# Hallucinated answers often introduce numbers not in source knowledge.
#--------------------------------------------------------------
def numeric_inconsistency(row):
    knowledge_nums = set(re.findall(r'\b\d+\b', str(row['knowledge'])))
    answer_nums = set(re.findall(r'\b\d+\b', str(row['answer'])))
    if not answer_nums:
        return 0
    inconsistent = answer_nums - knowledge_nums
    return len(inconsistent) / len(answer_nums)

df['numeric_inconsistency'] = df.apply(numeric_inconsistency, axis=1)

#--------------------------------------------------------------
# RESULTS: Signal strength by hallucination label
#--------------------------------------------------------------
print("=== Signal Averages by Hallucination Label ===")
print(df.groupby('hallucination')[['length_ratio', 'unknown_word_rate', 'qa_overlap', 'numeric_inconsistency']].mean())
print()

#--------------------------------------------------------------
# COMBINED HALLUCINATION SCORE (0 to 4)
# Each signal contributes 1 point if it exceeds its threshold
# Higher score = more likely to be a hallucination
#--------------------------------------------------------------
df['score'] = (
    (df['length_ratio'] > 0.1).astype(int) +
    (df['unknown_word_rate'] > 0.4).astype(int) +
    (df['qa_overlap'] > 0.2).astype(int) +
    (df['numeric_inconsistency'] > 0.5).astype(int)
)

print("=== Average Hallucination Score by Label ===")
print(df.groupby('hallucination')['score'].mean())
print()

print("=== Score Distribution by Hallucination Label ===")
print(df.groupby(['hallucination', 'score']).size().unstack(fill_value=0))
print()

#--------------------------------------------------------------
# SOFT FLAG: score >= 1 triggers a flag
# Philosophy: Better to over-flag and review than miss a hallucination
#--------------------------------------------------------------
df['hallucination_flag'] = (df['score'] >= 1).astype(int)
df['actual'] = (df['hallucination'] == 'yes').astype(int)
 
true_positive  = ((df['hallucination_flag'] == 1) & (df['actual'] == 1)).sum()
false_positive = ((df['hallucination_flag'] == 1) & (df['actual'] == 0)).sum()
false_negative = ((df['hallucination_flag'] == 0) & (df['actual'] == 1)).sum()
 
precision = true_positive / (true_positive + false_positive)
recall    = true_positive / (true_positive + false_negative)
 
print("=== Soft Flag Performance (score >= 1) ===")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")
print()

#--------------------------------------------------------------
# STRICT FLAG: score >= 3 triggers a flag
#--------------------------------------------------------------
df['strict_flag'] = (df['score'] >= 3).astype(int)
 
true_positive_s  = ((df['strict_flag'] == 1) & (df['actual'] == 1)).sum()
false_positive_s = ((df['strict_flag'] == 1) & (df['actual'] == 0)).sum()
false_negative_s = ((df['strict_flag'] == 0) & (df['actual'] == 1)).sum()
 
precision_s = true_positive_s / (true_positive_s + false_positive_s)
recall_s    = true_positive_s / (true_positive_s + false_negative_s)
 
print("=== Strict Flag Performance (score >= 3) ===")
print(f"Precision : {precision_s:.2f}")
print(f"Recall    : {recall_s:.2f}")
print("→ Use strict mode when human review capacity is limited.")
print()

#--------------------------------------------------------------
# REAL EXAMPLES FROM DATASET
#--------------------------------------------------------------
print("=== Example: Hallucinated Answers (score >= 2) ===")
hallucinated_examples = df[(df['actual'] == 1) & (df['score'] >= 2)][['question', 'answer', 'score']].head(2)
for _, row in hallucinated_examples.iterrows():
    print(f"Q: {row['question']}")
    print(f"A: {row['answer']}")
    print(f"Hallucination Score: {row['score']}/4")
    print()

print("=== Example: Non-Hallucinated Answers (score = 0) ===")
clean_examples = df[(df['actual'] == 0) & (df['score'] == 0)][['question', 'answer', 'score']].head(2)
for _, row in clean_examples.iterrows():
    print(f"Q: {row['question']}")
    print(f"A: {row['answer']}")
    print(f"Hallucination Score: {row['score']}/4")
    print()

#--------------------------------------------------------------
# SAVE OUTPUT
#--------------------------------------------------------------
df.to_csv("halueval_with_flags.csv", index=False)
print("Output saved to halueval_with_flags.csv")

