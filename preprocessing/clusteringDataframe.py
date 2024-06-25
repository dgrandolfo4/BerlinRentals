import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from amenities import amenities_matrix

def clustering_preprocessing(dataframe, cleaned_dataframe):
    """
    :param dataframe: output da utilizzare per il clustering
    :param cleaned_dataframe: dataframe pulito per creare le amenities
    :return: dataframe pronto per il clustering
    """

    # Creazione della matrice delle amenities
    amenities = amenities_matrix(cleaned_dataframe, (len(dataframe) / 3))

    # Rimozione delle colonne non numeriche e della colonna 'neighbourhood_cleansed'
    to_drop = list(dataframe.select_dtypes(['O']).columns) + ['neighbourhood_cleansed']
    dataframe.drop(to_drop, axis=1, inplace=True)
    dataframe = dataframe.reset_index()
    dataframe.drop('index', axis=1, inplace=True)

    # Resetta l'indice per evitare problemi di concatenazione
    amenities = amenities.reset_index()
    amenities.drop('index', axis=1, inplace=True)
    # Concatenazione del dataframe con la matrice delle amenities
    dataframe = pd.concat([dataframe, amenities], axis=1)
    
    # Imputazione dei valori mancanti con la strategia 'most_frequent'
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    dataframe = pd.DataFrame(imp_mean.fit_transform(dataframe), columns=dataframe.columns)
    
    # Normalizzazione dei dati con MinMaxScaler
    scaler = MinMaxScaler()
    dataframe = pd.DataFrame(scaler.fit_transform(dataframe), columns=dataframe.columns)
    
    # Applicazione del PCA
    pca = PCA(n_components=20)  # Seleziona il numero di componenti principali che desideri mantenere
    dataframe_pca = pca.fit_transform(dataframe)
    dataframe_pca = pd.DataFrame(dataframe_pca)
    
    # Arrotondamento dei valori a 10 decimali
    dataframe_pca = dataframe_pca.round(10)
    
    return dataframe_pca
