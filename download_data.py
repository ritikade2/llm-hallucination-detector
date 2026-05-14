from datasets import load_dataset
import pandas as pd

dataset = load_dataset("pminervini/HaluEval", "qa_samples")
df = pd.DataFrame(dataset["data"])
print(df.head())
print(df.columns)
df.to_csv("halueval_qa_samples.csv", index=False)
print("data saved")
