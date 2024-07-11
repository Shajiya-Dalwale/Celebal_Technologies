import pandas as pd
import numpy as np
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load Titanic dataset
def load_titanic():
    return pd.read_csv("F:\\Celebal\\Assignment_6\\Titanic-Dataset.csv")

def detect_lof_outliers(dataframe, col_names):
    imputer = SimpleImputer(strategy='mean')
    dataframe[col_names] = imputer.fit_transform(dataframe[col_names])
    
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    outliers = lof.fit_predict(dataframe[col_names])
    return dataframe[outliers == -1]

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * iqr
    up_limit = quartile3 + 1.5 * iqr
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)]

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]

def generalized_esd_test(dataframe, col_name, alpha=0.05, max_outliers=None):
    data = dataframe[col_name].dropna().values
    outliers = []
    n = len(data)
    k = 1
    while max_outliers is None or k <= max_outliers:
        z_scores = stats.zscore(data)
        max_z_score_idx = np.argmax(np.abs(z_scores))
        max_z_score = np.abs(z_scores[max_z_score_idx])
        threshold = stats.t.ppf(1 - alpha / (2 * n), n - 2) * (n - 1) / (n * (n - 2) ** 0.5)
        if max_z_score > threshold:
            outliers.append(max_z_score_idx)
            data = np.delete(data, max_z_score_idx)
            n -= 1
        else:
            break
        k += 1
    return outliers

df = load_titanic()

outliers_esd = generalized_esd_test(df, 'Age', max_outliers=10)
print(f"Outliers found using Generalized ESD Test in Age: {outliers_esd}")

outliers_lof = detect_lof_outliers(df, ['Fare', 'Age'])
print("Outliers found using Local Outlier Factor (LOF):")
print(outliers_lof.head())

sns.boxplot(x=df["Age"])
plt.title('Distribution of Age')
plt.show()
