import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def is_fraud(str):
    if 'fraud' in str:
        return 1
    else:
        return 0

if __name__ == '__main__':
    df = pd.read_csv('../../data/downsampled-df.csv')
    # df['fraud'] = df['acct_type'].apply(is_fraud)

    # plot unbalanced data
    # font = {'weight': 'bold', 'size': 16}
    # fig1, ax = plt.subplots()
    # fig1.subplots_adjust(bottom=0.2)
    # ax.bar(1, 13044 , .2)
    # ax.bar(2, 1293, .2)
    # ax.set_ylabel('Email Count', fontsize=14)
    # ax.set_title(" Count of Fraud vs. Not Fraud Emails", fontsize=14)
    # ax.set_xticks([1, 2])
    # ax.set_xticklabels(("Not Fraud", "Fraud"), fontsize=14)
    # plt.savefig('../images/unbalancedfraud')

    # plot balanced data after downsampling
    font = {'weight': 'bold', 'size': 16}
    fig1, ax = plt.subplots()
    fig1.subplots_adjust(bottom=0.2)
    ax.bar(1, 1293 , .2)
    ax.bar(2, 1293, .2)
    ax.set_ylabel('Email Count', fontsize=14)
    ax.set_title(" Count of Fraud vs. Not Fraud Emails - After Downsampling", fontsize=14)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(("Not Fraud", "Fraud"), fontsize=14)
    plt.savefig('../images/balancedfraud')
