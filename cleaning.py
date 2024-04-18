import pandas as pd
import numpy as np
from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN

def missing_drop(df, features_keep=[], message=True):
  '''
  Drop missing data in a way that minimalizes data loss
  '''
  df = df.copy()

  df.dropna(subset=features_keep, inplace=True)

  def generate_missing_table():
    df_results = pd.DataFrame(columns=['missing', 'if_drop_col', 'if_drop_rows'])
    for feat in df:
      missing = df[feat].isna().sum()
      if missing > 0:
        count_drop_col = df.drop(columns=[feat]).count().sum()
        count_drop_row = df.dropna(subset=[feat]).count().sum()
        df_results.loc[feat] = [missing, count_drop_col, count_drop_row]
    return df_results

  def findmax(df):
    max = 0
    for ind in df.index:
      for col in ['if_drop_col', 'if_drop_rows']:
        if df.loc[ind, col] > max:
          candidate = (ind, col)
          max = df.loc[ind, col]
    return candidate

  df_results = generate_missing_table()

  while df_results.shape[0] > 0:
    ind, col = findmax(df_results)
    if col == 'if_drop_col':
      if message: print(f'Dropping {ind} by column')
      df.drop(columns=[df_results.index[0]], inplace=True)
    elif col == 'if_drop_rows':
      if message: print(f'Dropping {ind} by rows')
      df.dropna(axis=0, subset=[ind], inplace=True)
    df_results = generate_missing_table()

  return df

def missing_impute_categorial(df, message=True):
  '''
  replace all missing categorial data with 'missing'
  '''
  df = df.copy()
  for feat in df.columns:
    if df[feat].isna().sum() > 0:
      if not pd.api.types.is_numeric_dtype(df[feat]):
        if message: print(f'Imputing {feat} by missing')
        df[feat] = df[feat].fillna(value='missing')
  return df

def bin_low_cat(df, features=[], message=True, cutoff=0.05, replace_with='Other'):
  '''
  Bin low count groups
  If a group has less than 5% of the total rows, replace that group with `Other`
  '''
  df = df.copy()

  if features == []: features = df.columns

  for feat in features:
    if feat in df.columns:
      if not pd.api.types.is_numeric_dtype(df[feat]):
        other_list = df[feat].value_counts()[df[feat].value_counts() / df.shape[0] < cutoff].index
        if other_list.shape[0] > 0 and message: print(f'Binning low count groups for {feat}')
        df.loc[df[feat].isin(other_list), feat] = replace_with

  return df

def missing_impute_numeric(df, iterative=False, message=True):
  '''
  Fill missing data with substitute values
  - To enable Iterative, make sure to dummy code the categorial features first 
  '''
  df = df.copy()

  if not iterative:
    for feat in df.columns:
      if df[feat].isna().sum() > 0:
        if pd.api.types.is_numeric_dtype(df[feat]):
          if message: print(f'Imputing {feat} by median')
          df[feat] = df[feat].fillna(value=df[feat].median())
  else:
    if message: print(f'Iteratively imputing')
    imp = IterativeImputer(max_iter=10)
    df = pd.DataFrame(imp.fit_transform(df), columns=df.columns, index=df.index)

  return df

def dummy_encode(df):
  '''
  Get the one-hot encoding of categorial features
  '''
  return pd.get_dummies(df, drop_first=True)

def scale(df):
  '''
  Scale the dataset
  Used MinMax Scaler to preserve one hot encoding
  '''
  return pd.DataFrame(MinMaxScaler().fit_transform(df),
                      index = df.index, columns = df.columns)

def outlier_clean(df, features=[], method='cutoff', message=True):
  '''
  Clean the outliers of each feature
  '''
  
  df = df.copy()

  if features == []: features = df.columns

  for feat in features:
    if feat in df.columns:
      if pd.api.types.is_numeric_dtype(df[feat]) and not all(df[feat].value_counts().index.isin([0,1])):
        # Tukey boxplot rule: < 1.5 * (Qt3 - Qt1) < is an outlier
        q1 = df[feat].quantile(0.25)
        q3 = df[feat].quantile(0.75)
        min = q1 - (1.5 * (q3 - q1))
        max = q3 + (1.5 * (q3 - q1))
        min_count = df.loc[df[feat]<min].shape[0]
        max_count = df.loc[df[feat]>max].shape[0]
        if message: print(f'{feat} has {min_count} values below min and {max_count} values above max')
        if min_count > 0 or max_count > 0:
          if method == 'filter':
            df = df[df[feat] > min]
            df = df[df[feat] < max]
          elif method == 'cutoff':
            df.loc[df[feat] < min, feat] = min
            df.loc[df[feat] > max, feat] = max
          elif method == 'median':
            df.loc[df[feat] < min, feat] = df[feat].median()
            df.loc[df[feat] > max, feat] = df[feat].median()
          else:
            df.loc[df[feat] < min, feat] = np.nan
            df.loc[df[feat] > max, feat] = np.nan
            # You can use iterative imputer with NaN values!
      else:
        print(f'{feat} is categorial / dummy and was ignored')

  return df

def outlier_clean_dbscan(df, drop_percent=0.02, min_samples=5, message=True):
  '''
  Clean outliers based on the DBSCAN clustering algorithm
  Requirements: (1) No missing data (2) One-hot encoded (3) Scaled
  '''

  df = df.copy()
  outliers_per_eps = []
  outliers = df.shape[0]
  eps = 0

  while outliers > 0:
    eps += 0.05
    db = DBSCAN(metric='manhattan', min_samples=min_samples, eps=eps).fit(df)
    outliers = np.count_nonzero(db.labels_ == -1)
    outliers_per_eps.append(outliers)
    if message: print(f'eps: {round(eps, 2)}, outliers: {outliers}, percent: {round(outliers/df.shape[0], 3)}')

  drops = min(outliers_per_eps, key=lambda x:abs(x-round(df.shape[0] * drop_percent)))
  eps = (outliers_per_eps.index(drops) + 1) * 0.05

  if message: print(f'using {eps} as the eps value, dropping {drops} values')

  db = DBSCAN(metric='manhattan', min_samples=min_samples, eps=eps).fit(df)
  df['outlier'] = db.labels_

  df = df[df['outlier'] != -1]
  df.drop(columns='outlier', inplace=True)

  return df

def skew_correct(df, feature, max_power=50, message=True):
  '''
  Try to address the skewness of the data
  If the final skew is still significant, consider converting to binary encoding
  '''
  df = df.copy()
  skew = df[feature].skew()
  if message: print(f'Starting Skew: {round(skew, 3)}')
  i = 1
  while round(skew, 2) !=0 and i <= max_power:
    i += 0.01
    if skew > 0:
      new = np.power(df[feature], 1/i)
      skew = new.skew()
    else:
      new = np.power(df[feature], i)
      skew = new.skew()
  if message: print(f'Final Skew: {round(skew, 3)}')

  return new