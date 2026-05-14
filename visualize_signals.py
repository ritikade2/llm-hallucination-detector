import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_csv("halueval_with_flags.csv")


fig , axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Hallucination Signal Distribution", fontsize = 16, fontweight = 'bold')

signals = [
    ('length_ratio', 'Length Ratio'),
    ('unknown_word_rate', 'Unknown Word Rate'),
    ('qa_overlap', 'Question Answer Overlap'),
    ('numeric_inconsistency', 'Numeric Inconsistency')
]

colors = {'yes': "#aa3046", 'no': "#469367"}

for ax, (col, title) in zip(axes.flatten(), signals):
    for label, group in df.groupby('hallucination'):
        group[col].plot.kde(ax = ax, label = label, color = colors[label], linewidth = 2)
        ax.set_title(title)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(['Hallucinated', 'Not Hallucinated'])
        ax.grid(True, alpha = 0.3)

plt.tight_layout()
plt.savefig("signal_distribution.png", dpi = 150, bbox_inches = 'tight')
print("Chart saved as 'signal_distribution.png'")