import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from datetime import datetime, date
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

def basic_visualization(df, features):
  '''
  draw the features along with time
  '''
  n = len(features)
  f, ax = plt.subplots(nrows=n, ncols=1, figsize=(15, 25))

  for i in range(0, n):
    sns.lineplot(x=df.Date, y=df[features[i]], ax=ax[i], color='blue')
    ax[i].set_ylabel(ylabel=features[i], fontsize=14)

  plt.show()

def missing_visualization(df):
  '''
  visualize the missing data
  '''
  f, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,5))
  sns.heatmap(df.T.isna(), cmap='Blues')
  ax.set_title('Fields with Missing Values', fontsize=16)
  for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
  plt.show()

def rolling_window_chart(df, features, rolling_window):
  '''
  perform a rolling window analysis on the mean and std value
  '''
  n = len(features)
  f, ax = plt.subplots(nrows = n, ncols = 1, figsize=(15, 12))

  for i in range(0, n):
    sns.lineplot(x=df.Date, y=df[features[i]], ax=ax[i], color='red')
    sns.lineplot(x=df.Date, y=df[features[i]].rolling(rolling_window).mean(), ax=ax[i], color='black', label='rolling mean')
    sns.lineplot(x=df.Date, y=df[features[i]].rolling(rolling_window).std(), ax=ax[i], color='blue', label='rolling std')
    ax[i].set_ylabel(ylabel=features[i], fontsize=14)

  plt.show()

def adf(df, features):
  '''
  Generate the augmented dickey-fuller test statistics
  '''
  results = pd.DataFrame(columns = ['ADF Statistic', 'p-value', '5% Critical Value'])
  for feat in features:
    result = adfuller(df[feat].values)
    adf_stat = result[0]
    p = result[1]
    crit_val_5 = result[4]['5%']
    results.loc[feat] = [adf_stat, p, crit_val_5]

  return results.round(3)

def add_time_features(df):
  '''
  Generate time features
  '''
  df = df.copy()

  df['year'] = pd.DatetimeIndex(df['Date']).year
  df['month'] = pd.DatetimeIndex(df['Date']).month
  df['day'] = pd.DatetimeIndex(df['Date']).day
  df['day_of_year'] = pd.DatetimeIndex(df['Date']).dayofyear
  df['quarter'] = pd.DatetimeIndex(df['Date']).quarter
  df['season'] = df.month%12 // 3 + 1

  df[['Date', 'year', 'month', 'day', 'day_of_year', 'quarter', 'season']].head()

  return df

def convert_cyclical(df):
  '''
  convert cyclical features into smooth circles
  '''
  df = df.copy()

  month_in_year = 12
  df['month_sin'] = np.sin(2*np.pi*df.month/month_in_year)
  df['month_cos'] = np.cos(2*np.pi*df.month/month_in_year)

  days_in_month = 30
  df['day_sin'] = np.sin(2*np.pi*df.day/days_in_month)
  df['day_cos'] = np.cos(2*np.pi*df.day/days_in_month)

  days_in_year = 365
  df['day_of_year_sin'] = np.sin(2*np.pi*df.day_of_year/days_in_year)
  df['day_of_year_cos'] = np.cos(2*np.pi*df.day_of_year/days_in_year)

  weeks_in_year = 52.1429
  df['week_of_year_sin'] = np.sin(2*np.pi*df.week_of_year/weeks_in_year)
  df['week_of_year_cos'] = np.cos(2*np.pi*df.week_of_year/weeks_in_year)

  quarters_in_year = 4
  df['quarter_sin'] = np.sin(2*np.pi*df.quarter/quarters_in_year)
  df['quarter_cos'] = np.cos(2*np.pi*df.quarter/quarters_in_year)

  seasons_in_year = 4
  df['season_sin'] = np.sin(2*np.pi*df.season/seasons_in_year)
  df['season_cos'] = np.cos(2*np.pi*df.season/seasons_in_year)

  return df

def decomposition(df, features, period):
  '''
  decompose each feature into trend, season, and residue
  '''
  n = len(features)
  f, ax = plt.subplots(ncols = 4, nrows = n, figsize = (16, 8))
  for i in range(0, n):
    decomp = seasonal_decompose(df[features[i]], period=period, model='additive', extrapolate_trend='freq')
    sns.lineplot(x=df.Date, y=decomp.observed, ax=ax[i][0])
    ax[i][0].set_ylabel(f'{features[i]}', fontsize=14)
    sns.lineplot(x=df.Date, y=pd.Series.to_list(decomp.trend), ax=ax[i][1])
    sns.lineplot(x=df.Date, y=pd.Series.to_list(decomp.seasonal), ax=ax[i][2])
    sns.lineplot(x=df.Date, y=pd.Series.to_list(decomp.resid), ax=ax[i][3])

  ax[0][0].set_title('Observed', fontsize=14)
  ax[0][1].set_title('Trend', fontsize=14)
  ax[0][2].set_title('Season', fontsize=14)
  ax[0][3].set_title('Residue', fontsize=14)
  plt.show()

def ts_train_test_split_visualization(folds, X, y, N_SPLITS):
  '''
  Visualize the rolling window / expanding window
  '''

  f, ax = plt.subplots(nrows=N_SPLITS, ncols=2, figsize=(16, 9))

  for i, (train_index, valid_index) in enumerate(folds.split(X)):
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]

    sns.lineplot(x= X_train, y= y_train, ax=ax[i,0], color='dodgerblue', label='train')
    sns.lineplot(x= X_train[len(X_train) - len(X_valid):(len(X_train))], 
                  y= y_train[len(X_train) - len(X_valid):(len(X_train))], 
                  ax=ax[i,1], color='dodgerblue', label='train')

    for j in range(2):
        sns.lineplot(x= X_valid, y= y_valid, ax=ax[i, j], color='darkorange', label='validation')
    ax[i, 0].set_title(f"Rolling Window (Split {i+1})", fontsize=16)
    ax[i, 1].set_title(f"Expanding Window (Split {i+1})", fontsize=16)

  for i in range(N_SPLITS):
    ax[i, 0].set_xlim([X[0], X[len(X)-1]])
    ax[i, 1].set_xlim([X[0], X[len(X)-1]])

  plt.tight_layout()
  plt.show()

def plot_approach_evaluation(X, y, X_test, y_test, y_pred, folds, score_mae, score_rsme, approach_name):
    '''
    Draw the evaluation charts
    '''
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    f.suptitle(approach_name, fontsize=16)
    sns.lineplot(x=X.Date, y=y, ax=ax[0], color='dodgerblue', label='Training', linewidth=2)
    sns.lineplot(x=X_test.Date, y=y_test, ax=ax[0], color='gold', label='Ground Truth', linewidth=2) #navajowhite
    sns.lineplot(x=X_test.Date, y=y_pred, ax=ax[0], color='darkorange', label='Predicted', linewidth=2)
    ax[0].set_xlim([date(2018, 6, 30), date(2020, 6, 30)])
    ax[0].set_ylim([-27, -23])
    ax[0].set_title(f'Prediction \n MAE: {mean_absolute_error(y_test, y_pred):.2f}, RSME: {math.sqrt(mean_squared_error(y_test, y_pred)):.2f}', fontsize=14)
    ax[0].set_xlabel(xlabel='Date', fontsize=14)
    ax[0].set_ylabel(ylabel='Depth to Groundwater P25', fontsize=14)

    sns.lineplot(x=folds, y=score_mae,  color='gold', label='MAE', ax=ax[1]),
    sns.lineplot(x=folds, y=score_rsme, color='red', label='RSME', ax=ax[1])
    ax[1].set_title('Loss', fontsize=14)
    ax[1].set_xlabel(xlabel='Fold', fontsize=14)
    ax[1].set_ylabel(ylabel='Loss', fontsize=14)
    ax[1].set_ylim([0, 4])   
    plt.show()