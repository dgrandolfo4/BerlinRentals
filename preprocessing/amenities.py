import re
import pandas as pd
import numpy as np
from collections import Counter

def splitting_amenities(element):
    """
    Suddivide una stringa di comfort separate da virgole in una lista di comfort.
    
    Args:
        element (str): La stringa di comfort.

    Returns:
        list: Una lista di comfort.
    """
    list_of_amenities = element.split(',')
    list_of_amenities = [word.lower() for word in list_of_amenities]
    return list_of_amenities

def amenities_matrix(dataframe, treshold):
    """
    Crea una matrice di booleani rappresentante la presenza o l'assenza di determinate comfort nei dati del dataframe.

    Args:
        dataframe (pd.DataFrame): Il dataframe contenente i dati degli annunci immobiliari, con la colonna 'amenities'.
        treshold (int): La soglia per filtrare le comfort meno comuni.

    Returns:
        pd.DataFrame: Una matrice di booleani rappresentante la presenza o l'assenza delle comfort.
    """
    dataframe['amenities'] = dataframe['amenities'].apply(lambda x: x[1:-1])
    dataframe['amenities'] = dataframe['amenities'].apply(lambda x: x.replace(', ', ','))
    dataframe['amenities'] = dataframe['amenities'].apply(lambda x: x.replace(' , ', ','))
    dataframe['amenities'] = dataframe['amenities'].apply(lambda x: re.sub(r'(?<=[a-zA-Z0-9])[,](?=[a-zA-Z0-9])', ' ', x))

    # Elimina comfort che non superano la soglia
    w_count = Counter()
    dataframe['amenities'].str.lower().str.split(',').apply(w_count.update)
    lista = []
    for amenities in w_count:
        if w_count[amenities] <= treshold:
            lista.append(amenities)
    for element in lista:
        del w_count[element]
    bow = list(w_count.keys())

    columnIndex = dataframe.columns.get_loc('amenities')
    to_add_row = []
    index = 0
    indexAmenities = {}

    for i in range(len(dataframe.amenities)):
        element = dataframe.iloc[i, columnIndex]
        list_of_amenities = splitting_amenities(element)
        list_of_amenities = list(set(list_of_amenities) & set(bow))
        to_add_row.append(list_of_amenities)
        for word in list_of_amenities:
            if word not in indexAmenities:
                indexAmenities[word] = index
                index += 1

    rows = len(dataframe.amenities)
    columns = len(indexAmenities)

    amenities_matrix = np.zeros((rows, columns))
    i = 0
    for row in to_add_row:
        amenities_matrix[i, :] = writer(row, columns, indexAmenities)
        i += 1

    # Definisce un dataframe a partire dalla matrice
    amenities_dataframe = pd.DataFrame(amenities_matrix, columns=list(indexAmenities.keys())).astype("boolean")
    amenities_dataframe.columns = [cleaning_column(text) for text in list(amenities_dataframe.columns)]

    return amenities_dataframe


def writer(row, columns, indexes):
    """
    Scrive i valori della riga corrente nella matrice.

    Args:
        row (list): Lista dei valori presenti nella riga.
        columns (int): Numero di colonne della matrice.
        indexes (dict): Dizionario contenente gli indici delle colonne.

    Returns:
        np.array: Array con i valori della riga corrente.
    """
    written_columns = np.zeros(columns)
    for value in row:
        idx = indexes[value]
        written_columns[idx] = 1
    return written_columns


def cleaning_column(e):
    """
    Pulisce il nome della colonna.

    Args:
        e (str): Nome della colonna da pulire.

    Returns:
        str: Nome della colonna pulito.
    """
    e = re.sub(r'[\s+]]*', '_', e)
    e = re.sub(r'[\"]', '', e)
    return e
