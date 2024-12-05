import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from run_clustering_methods_and_classifiers.LouvainAlgorithm import louvain_algorithm
from run_clustering_methods_and_classifiers.kmeans import kmeans
from run_clustering_methods_and_classifiers.DBSCAN import dbscan
from run_clustering_methods_and_classifiers.log_reg_classifier import logistic_regression_classifier
from run_clustering_methods_and_classifiers.SpectralNet import spectral_net
from run_clustering_methods_and_classifiers.spectral_clustering import spectral_clustering


def main():
    X_train, X_test, y_train, y_test = prepare_data()

    test_louvain(X_train, X_test, y_train, y_test)
    test_kmeans(X_train, X_test, y_train, y_test)
    test_dbscan(X_train, X_test, y_train, y_test)
    test_spectral_net(X_train, X_test, y_train, y_test)



def test_louvain(X_train, X_test, y_train, y_test):
    train_cluster_distances_df, test_cluster_distances_df = louvain_algorithm(X_train, X_test)
    # Add the new features to the original dataframes
    X_train = pd.concat([X_train, train_cluster_distances_df], axis=1)
    X_test = pd.concat([X_test, test_cluster_distances_df], axis=1)

    # Save all the data as CSV
    X_train.to_csv('X_train_louvain.csv', index=False)
    X_test.to_csv('X_test_louvain.csv', index=False)

    # Run log_reg_classifier
    results = logistic_regression_classifier(X_train, X_test, y_train, y_test)

    # Save the results as a CSV
    results.to_csv('results_louvain.csv', index=False)



def test_kmeans(X_train, X_test, y_train, y_test):
    train_clusters, test_clusters = kmeans(X_train, X_test)
    # Add the new features to the original dataframes
    X_train = pd.concat([X_train, train_clusters], axis=1)
    X_test = pd.concat([X_test, test_clusters], axis=1)

    # Save all the data as CSV
    X_train.to_csv('X_train_kmeans.csv', index=False)
    X_test.to_csv('X_test_kmeans.csv', index=False)

    # Run log_reg_classifier
    results = logistic_regression_classifier(X_train, X_test, y_train, y_test)

    # Save the results as a CSV
    results.to_csv('results_kmeans.csv', index=False)



def test_dbscan(X_train, X_test, y_train, y_test):
    train_clusters, test_clusters = dbscan(X_train, X_test)
    # Add the new features to the original dataframes
    X_train = pd.concat([X_train, train_clusters], axis=1)
    X_test = pd.concat([X_test, test_clusters], axis=1)

    # Save all the data as CSV
    X_train.to_csv('X_train_dbscan.csv', index=False)
    X_test.to_csv('X_test_dbscan.csv', index=False)

    # Run log_reg_classifier
    results = logistic_regression_classifier(X_train, X_test, y_train, y_test)

    # Save the results as a CSV
    results.to_csv('results_dbscan.csv', index=False)



def test_spectral_net(X_train, X_test, y_train, y_test):
    train_clusters, test_clusters = spectral_net(X_train, X_test)
    # Add the new features to the original dataframes
    X_train = pd.concat([X_train, train_clusters], axis=1)
    X_test = pd.concat([X_test, test_clusters], axis=1)

    # Save all the data as CSV
    X_train.to_csv('X_train_spectral_net.csv', index=False)
    X_test.to_csv('X_test_spectral_net.csv', index=False)

    # Run log_reg_classifier
    results = logistic_regression_classifier(X_train, X_test, y_train, y_test)

    # Save the results as a CSV
    results.to_csv('results_spectral_net.csv', index=False)



def test_spectral_clustering(X_train, X_test, y_train, y_test):
    train_clusters, test_clusters = spectral_clustering(X_train, X_test)
    # Add the new features to the original dataframes
    X_train = pd.concat([X_train, train_clusters], axis=1)
    X_test = pd.concat([X_test, test_clusters], axis=1)

    # Save all the data as CSV
    X_train.to_csv('X_train_spectral_clustering.csv', index=False)
    X_test.to_csv('X_test_spectral_clustering.csv', index=False)

    # Run log_reg_classifier
    results = logistic_regression_classifier(X_train, X_test, y_train, y_test)

    # Save the results as a CSV
    results.to_csv('results_spectral_clustering.csv', index=False)


def prepare_data():
    df = pd.read_csv('creditcard.csv')

    X = df.drop(['Time', 'Class'], axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    main()

