import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from statsmodels.stats.multicomp import MultiComparison

def corr(df):
  '''
  Measure and plot the correlation between all numeric values
  For two values with high colinearity, the options are:
  - (1) drop one of the features
  - (2) do principle component analysis
  '''

  fig = plt.figure()
  sns.heatmap(df.corr(numeric_only=True), annot=True)
  plt.show()

def anova(df, feat_cat, feat_num):
  '''
  Compute the oneway anova for a numeric feature and a categorial feature
  - A high anova value means the categorial feature significantly distinguishes the numeric feature
  '''
  groups = df[feat_cat].unique()
  df_grouped = df.groupby(feat_cat)
  group_labels = []
  for g in groups:
    g_list = df_grouped.get_group(g)
    group_labels.append(g_list[feat_num])

  f, p = stats.f_oneway(*group_labels)

  return (f, p)

def bar_chart(df, feat_cat, feat_num, f, p):
  '''
  Plot the bar chart between a categorial value and a numerical value
  along with anova test results
  '''

  sns.barplot(df, x=feat_cat, y=feat_num)

  textstr = f'F: {round(f, 3)}\n'
  textstr += f'p: {round(p, 3)}'

  plt.text(0, 0, textstr, fontsize=12, transform=plt.gcf().transFigure)
  plt.show()

def crosstab(df, feature, target, X2, p, contingency_table):
  '''
  Plot the contingency table between two categorial values
  along with Pearson Chi_2 test results
  '''
  crosstab = pd.crosstab(df[feature], df[target])
  ct_df = pd.DataFrame(contingency_table, columns=crosstab.columns, index=crosstab.index)
  sns.heatmap(ct_df, annot=True)

  textstr = f'X2: {round(X2,2)}\n'
  textstr += f'p: {round(p,2)}'

  plt.text(0, 0, textstr, fontsize=12, transform=plt.gcf().transFigure)
  plt.show()

def crosstab_bar(df, feature, target, X2, p):
  '''
  Plot the crosstab bar chart between two categorial values
  along with Pearson Chi_2 test results
  '''
  crosstab = pd.crosstab(df[feature], df[target])
  crosstab.plot(kind='bar', stacked=True)

  textstr = f'X2: {round(X2,2)}\n'
  textstr += f'p: {round(p,2)}'

  plt.text(0, 0, textstr, fontsize=12, transform=plt.gcf().transFigure)
  plt.show()

def scatterplot(df, feature, target, r, p):
  '''
  Plot the scatter plot between two numerical values
  along with Pearson R test results
  '''
  textstr = 'r = ' + str(round(r, 2)) + '\n'
  textstr += 'p = ' + str(round(p, 2)) + '\n'
  textstr += str(feature) + ' skew = ' + str(round(df[feature].skew(), 2)) + '\n'
  textstr += str(target) + ' skew = ' + str(round(df[target].skew(), 2)) + '\n'
  ax = sns.jointplot(data=df, x=feature, y=target, kind='reg')
  ax.figure.text(1, 0, textstr, fontsize=12)
  plt.show()

def bivstats(df, target, features=[], plot=False, tukey=False):
  '''
  Provide bivariate statistics between features and the target
  plot: whether to make corresponding plots
  tukey: whether to print the tukey pairwise test table
  '''
  output_df = pd.DataFrame(columns=['|R|', 'F', 'X2', 'p-Value'])

  if features == []: features = df.columns

  for feat in features:
    if feat != target:
      df_temp = df[[feat, target]].copy().dropna(axis = 0)
      if pd.api.types.is_numeric_dtype(df_temp[feat]) and pd.api.types.is_numeric_dtype(df_temp[target]):
        r, p = stats.pearsonr(df_temp[feat], df_temp[target])
        output_df.loc[feat] = [abs(round(r, 3)), np.nan, np.nan, round(p, 3)]
        if plot: scatterplot(df_temp, feat, target, r, p)
      elif (not pd.api.types.is_numeric_dtype(df_temp[feat])) and pd.api.types.is_numeric_dtype(df_temp[target]):
        f, p = anova(df_temp, feat, target)
        output_df.loc[feat] = [np.nan, round(f,3), np.nan, round(p,3)]
        if plot: bar_chart(df_temp, target, feat, f, p)
        if tukey:
          print(feat)
          print(MultiComparison(df_temp[target], df_temp[feat]).tukeyhsd())
          print('\n')
      elif (not pd.api.types.is_numeric_dtype(df_temp[target])) and pd.api.types.is_numeric_dtype(df_temp[feat]):
        f, p = anova(df_temp, target, feat)
        output_df.loc[feat] = [np.nan, round(f,3), np.nan, round(p,3)]
        if plot: bar_chart(df_temp, target, feat, p, f)
      else:
        contingency_table = pd.crosstab(df_temp[feat], df_temp[target])
        X2, p, dof, ct = stats.chi2_contingency(contingency_table)
        output_df.loc[feat] = [np.nan, np.nan, round(X2, 3), round(p,3)]
        if plot: crosstab_bar(df, feat, target, X2, p, ct)

  return output_df.sort_values(by=['|R|', 'F', 'X2'], ascending=False)