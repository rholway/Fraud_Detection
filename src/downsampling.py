import pandas as pd
import numpy as np
from sklearn.utils import resample


if __name__ == '__main__':
    df = pd.read_csv('../../data/data2.csv')

    # due to class imbalance, I'm going to downsample
    # first, separate majority and minority classes
    df_not_fraud = df[df.fraud==0]
    df_fraud = df[df.fraud==1]

    # Downsample majority class
    df_not_fraud_downsampled = resample(df_not_fraud,
                                     replace=False,    # sample without replacement
                                     n_samples=1293,     # to match minority class
                                     random_state=123) # reproducible results

    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_not_fraud_downsampled, df_fraud])

    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df_downsampled.to_csv('../../data/downsampled-df.csv')
