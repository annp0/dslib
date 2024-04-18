import pandas as pd
import datetime as dt
import math

def parse_date(df, time_features=[], days_since_today=False):
  '''
  Parse the time and date
  time_features: columns to be interpreted as time
  days_since_today: whether to create a `day_since_today` feature
  '''
  df = df.copy()

  for feat in time_features:
    df[feat] = pd.to_datetime(df[feat])
    df[f'{feat}_year'] = df[feat].dt.year
    df[f'{feat}_month'] = df[feat].dt.month
    df[f'{feat}_day'] = df[feat].dt.day
    df[f'{feat}_weekday'] = df[feat].dt.day_name()

  if days_since_today:
    df[f'{feat}_days_until_today'] = (dt.today() - df[feat]).dt.days

  df.drop(columns=time_features, inplace=True)

  return df

def basic_wrangling(df, message=True):
  '''
  Basic data wrangling
  The following will be eliminated:
  - Columns and rows with too many values missing
  - Categorial and Integer columns with 70% unique values
  - Columns with single values
  '''
  df = df.copy()
  rows_old, cols = df.shape
  df.dropna(axis=0, thresh=math.floor(0.3 * cols))
  rows = df.shape[0]
  if message: print(f'Dropped {rows-rows_old} rows')
  for feat in df.columns:
    missing = df[feat].isna().sum()
    unique = df[feat].nunique()
    if missing / rows >= 0.7:
      if message: print(f'Too many missing values {round(missing/rows,2)} for {feat}')
      df.drop(columns=[feat], inplace=True)
    elif unique / rows >= 0.7 and (df[feat].dtype in ['int64', 'object']):
      if message: print(f'Too many unique values {round(unique/rows,2)} for {feat}')
      df.drop(columns=[feat], inplace=True)
    elif unique == 1:
      if message : print(f'Only one value for {feat}')
      df.drop(columns=[feat], inplace=True)
  return df