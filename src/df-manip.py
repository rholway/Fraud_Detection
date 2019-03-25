import pandas as pd
import numpy as np
from bs4 import BeautifulSoup


def is_fraud(str):
    if 'fraud' in str:
        return 1
    else:
        return 0

def parse_html(str):
    soup = BeautifulSoup(str, "lxml")
    return soup.get_text()


if __name__ == '__main__':
    df = pd.read_json('../../data/data.json')
    df['fraud'] = df['acct_type'].apply(is_fraud)

    # create new df with only description and targets
    df2 = df.filter(['description', 'fraud'], axis = 1)
    df2['description'] = df2['description'].apply(parse_html)

    df2.to_csv('../../data/data2.csv')
