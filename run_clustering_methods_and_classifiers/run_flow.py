import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from run_clustering_methods_and_classifiers.LouvainAlgorithm import louvain_algorithm
from run_clustering_methods_and_classifiers.kmeans import kmeans_algorithm

def main():
    X_train, X_test, y_train, y_test = prepare_data()

def run_clustering_methods(X_train, X_test):
    louvain_algorithm(X_train, X_test)
    kmeans_algorithm(X_train, X_test)

def prepare_data():
    df = pd.read_csv('creditcard.csv')

    X = df.drop(['Class'], axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    main()

