from sys import argv

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from sklearn.cluster import KMeans


def k_cluster(dataframe,k,max):
    """
    Crea i cluster utilizzando l'algoritmo K-Means.

    :param dataframe: Dataframe con le features da utilizzare per il clustering.
    :type dataframe: pd.DataFrame
    :param k: Numero di cluster.
    :type k: int
    :param max_it: Numero massimo di iterazioni.
    :type max_it: int
    :return: Etichette dei cluster assegnate a ciascuna riga del dataframe.
    :rtype: array
    """
    print("Creazione clusters...")

    km = KMeans(n_clusters=k, max_iter=max, n_init=10)
    km.fit(dataframe)
    clusters = km.fit_predict(dataframe)
    return clusters

def calculate_optimal_clusters(df, max_clusters=10):
    """
    Calcola il numero ottimale di clustres tramite il metodo del gomito.

    :param dataframe: Dataframe con le features da utilizzare per il clustering.
    :type dataframe: pd.DataFrame
    :param max_clusters: Numero di cluster.
    :type max_clusters: int
    :return: Numero ottimale di clusters.
    :rtype: int
    """
    # Assicurati di selezionare solo le colonne numeriche per il clustering
    numeric_df = df.select_dtypes(include=[np.number])

    # Lista per memorizzare i valori di SSE per ogni numero di cluster
    sse = []

    # Calcolo della somma degli errori quadratici (SSE) per ogni numero di cluster
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(numeric_df)
        sse.append(kmeans.inertia_)

    # Chiedi all'utente se vuole visualizzare il grafico finché non risponde correttamente
    while True:
        show_plot = input("Vuoi visualizzare il grafico del metodo del gomito? (y/n): ").strip().lower()
        if show_plot in ['y', 'n']:
            break
        else:
            print("Risposta non valida. Per favore rispondi con 'y' o 'n'.")

    # Mostra il grafico solo se l'utente risponde 'y'
    if show_plot == 'y':
        # Plot del metodo del gomito
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_clusters + 1), sse, marker='o')
        plt.suptitle("Elbow Method Graph")
        plt.title('Metodo del gomito')
        plt.xlabel('Numero di cluster')
        plt.ylabel('Somma degli errori quadratici (SSE)')
        plt.show()

    # Calcolo del numero ottimale di cluster usando il metodo del gomito
    optimal_clusters = np.argmin(np.diff(sse, 2)) + 2  # +2 perché np.diff riduce di 2 il numero di punti

    return optimal_clusters

def main():
    try:
        print("Clustering in corso...")
        dataframe = pd.read_csv('./datasets/cleaned_dataset.csv') # cleaned dataset
        iterations = int(argv[1]) if len(sys.argv) > 1 else 10 # number of max iterations
        k = calculate_optimal_clusters(dataframe, max_clusters=10) # number of optimal clusters (calculated with the Elbow Method)
        cluster_centroids = k_cluster(dataframe, k, iterations)

        df_prolog = pd.read_csv('./datasets/prolog_dataframe.csv')
        df_prolog['cluster'] = cluster_centroids
        df_prolog.to_csv('./datasets/prolog_dataframe.csv', index = False)
        print("Clustering eseguito.")

    except FileNotFoundError as e:
        print("Cleaned Dataset not found", e)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()