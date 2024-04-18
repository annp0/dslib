import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def unistats(df):
  '''
  Provide a basic statistics table for each variable
  '''
  output_df = pd.DataFrame(columns=['Count', 'Missing', 'Unique', 'Dtype', 'Mode', 'Min',
                                    'Median', 'Mean', 'Max', 'Std', 'Skew'])

  for col in df:
    if pd.api.types.is_numeric_dtype(df[col]):
      output_df.loc[col] = [df[col].count(), df[col].isnull().sum(),
                            df[col].nunique(), df[col].dtype, df[col].mode()[0],
                            df[col].min(), df[col].median(), df[col].mean(), df[col].max(),
                            df[col].std(), df[col].skew()]
    else:
      output_df.loc[col] = [df[col].count(), df[col].isnull().sum(), df[col].nunique(),
                            df[col].dtype, df[col].mode()[0], '', '', '', '', '', '']

  return output_df

def uniplot(df, features=[]):
  '''
  Plot the histograms of single variables to determine:
  - for numerical values: skewness problems
  - for categorial values: class imbalance problems
  '''
  if features == []: features = df.columns

  for feat in features:
    sns.histplot(df[feat].dropna())
    plt.show()

def uniboxplot(df, features=[]):
  '''
  Plot the outliers for each single numeric value
  '''
  if features == []: features = df.columns

  for feat in features:
    if pd.api.types.is_numeric_dtype(df[feat]):
      fig = plt.figure(figsize=(3,5))
      sns.boxplot(df[feat])
      plt.show()