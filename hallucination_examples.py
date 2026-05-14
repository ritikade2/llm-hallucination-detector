
import pandas as pd
df = pd.read_csv('halueval_with_flags.csv')
df['actual'] = (df['hallucination'] == 'yes').astype(int)
print('HALLUCINATED (score 3+):')
print(df[(df['actual']==1) & (df['score']>=3)][['question','answer','score']].iloc[2:5].to_string())
print()
print('NOT HALLUCINATED (score 0):')
print(df[(df['actual']==0) & (df['score']==0)][['question','answer','score']].iloc[2:5].to_string())
