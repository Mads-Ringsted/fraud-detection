import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def main():
    X_train, X_test, y_train, y_test = prepare_data()



def prepare_data():
    df = pd.read_csv('creditcard.csv')

    X = df.drop(['Class'], axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    main()

