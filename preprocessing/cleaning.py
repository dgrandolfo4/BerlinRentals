import pandas as pd
import re
from sklearn.impute import SimpleImputer
from bayesianDataframe import datasetBuilding
from clusteringDataframe import *

def is_center(column,center):
    """
    Permette di avere una verifica sulla posizione centrale del quartiere.

    :param column: Quartiere da verificare.
    :type column: str
    :param center: Lista di quartieri centrali a Berlino.
    :type center: list
    :return: True se il quartiere è nel centro, False altrimenti.
    :rtype: bool
    """

    for i in range(0,len(center)):
        if center[i] == column:
            return True
    return False

def cleaning_dataset(dataframe):
    """
    Esegue il pre-processing del dataset.
    Rimuove alcune colonne, gestisce i dati mancanti, effettua trasformazioni su alcune colonne, e applica altre operazioni di pulizia.

    :param dataframe: Il dataset da pulire.
    :type dataframe: pd.DataFrame
    :return: Il dataframe pulito e pronto per essere usato.
    :rtype: pd.DataFrame
    """
    print("Cleaning...")

    dataframe = dataframe.drop(["listing_url", "scrape_id", "last_scraped", "neighborhood_overview",
                          "picture_url", "host_id", "host_url", "host_name", "host_since", "host_location",
                          "host_about", "host_thumbnail_url", "host_picture_url", "host_neighbourhood",
                          "neighbourhood_group_cleansed", "latitude", "longitude", "bathrooms",
                          "minimum_minimum_nights", "maximum_minimum_nights",
                          "minimum_maximum_nights", "maximum_maximum_nights", "minimum_nights_avg_ntm",
                          "maximum_nights_avg_ntm", "calendar_updated", "availability_30",
                          "availability_60", "availability_90", "availability_365", "calendar_last_scraped",
                          "number_of_reviews_ltm", "number_of_reviews_l30d", "first_review", "last_review", "license",
                          "calculated_host_listings_count", "calculated_host_listings_count_entire_homes",
                          "calculated_host_listings_count_private_rooms",
                          "calculated_host_listings_count_shared_rooms", "reviews_per_month","host_has_profile_pic",
                          "host_acceptance_rate","host_listings_count","host_total_listings_count","review_scores_accuracy",
                          "review_scores_checkin","review_scores_communication","description","name"
                          ], axis=1)
    dataframe = dataframe[dataframe.property_type.isin(['Entire apartment', 'Private room in apartment',
                                               'Private room in house', 'Private room in townhouse',
                                               'Entire condominium', 'Entire house', 'Entire loft',
                                               'Entire townhouse', 'Entire rental unit'])]

    dataframe = dataframe.drop(dataframe[dataframe.bedrooms.isnull()].index)
    dataframe["bedrooms"] = dataframe["bedrooms"].astype('int')

    dataframe['property_type'] = dataframe.property_type.apply(lambda c: re.sub(' ', '_', c))
    dataframe['room_type'] = dataframe.room_type.apply(lambda c: re.sub(' ', '_', c))

    dataframe['neighbourhood_cleansed'] = dataframe['neighbourhood_cleansed'].str.replace(' / ', '/')
    dataframe['neighbourhood_cleansed'] = dataframe['neighbourhood_cleansed'].str.replace('  ', ' ')
    dataframe["neighbourhood_cleansed"] = dataframe["neighbourhood_cleansed"].apply(lambda c: re.sub('[\\s][-][\\s][\\S]+', '', c))
    dataframe["neighbourhood_cleansed"] = dataframe["neighbourhood_cleansed"].apply(lambda c: re.sub("[']", '', c)).astype('category')

    dataframe['price'] = dataframe.price.apply(lambda c: re.sub('[$,]', '', str(c))).astype('float')

    # nell'attributo beds completo le celle vuote con la media e casto ad int
    dataframe["beds"] = SimpleImputer(strategy='median').fit_transform(dataframe[["beds"]]).round()
    dataframe["beds"] = dataframe["beds"].astype('int')

    # nell'attributo bathrooms_test completo le celle vuote con la media arrotondandole
    most_frequent = SimpleImputer(strategy='most_frequent')
    dataframe.loc[:, 'bathrooms_text'] = most_frequent.fit_transform(dataframe[['bathrooms_text']])
    dataframe["bathrooms_text"] = dataframe.bathrooms_text.apply(lambda c : re.sub('[a-zA-Z]*', '', c)).astype('float').round()

    #discretizzazione sulle boolean
    dataframe['host_is_superhost'] = dataframe.host_is_superhost.apply(lambda c: 1 if c == 't' else 0)
    dataframe['has_availability'] = dataframe.has_availability.apply(lambda c: 1 if c == 't' else 0)
    dataframe['instant_bookable'] = dataframe.instant_bookable.apply(lambda c: 1 if c == 't' else 0)
    dataframe['host_identity_verified'] = dataframe.host_identity_verified.apply(lambda c: 1 if c == 't' else 0)

    # inserisco la media nelle celle vuote dell'attributo review_score_rating
    dataframe["review_scores_rating"] = SimpleImputer(strategy='median').fit_transform(dataframe[["review_scores_rating"]])

    # riutilizziamo una colonna in cui inseriamo valore booleani per capire se un hotel è vicino o no al centro.
    dataframe = dataframe.rename(columns={'neighbourhood': 'is_center'})

    # quartieri che si trovono in centro a Berlino
    center = ["Alexanderplatz", "Brunnenstr. Süd", "Moabit Ost", "Regierungsviertel", "Tiergarten Süd", "Südliche Friedrichstadt", "Tempelhofer Vorstadt", "Tiergarten Süd"]
    dataframe["is_center"] = dataframe.neighbourhood_cleansed.apply(lambda c: 1 if is_center(c,center) else 0).astype('int')

    # inserisco la media nel host_response_rate
    dataframe["host_response_rate"] = dataframe.host_response_rate.astype('category')
    dataframe["host_response_rate"] = dataframe.host_response_rate.apply(lambda c: re.sub('[%]','',c))
    dataframe["host_response_rate"] = SimpleImputer(strategy='median').fit_transform(dataframe[["host_response_rate"]])
    print("Cleaning eseguito.")
    return dataframe

def main():
    try:
        print("Avvio pre processing...")
        dataframe = pd.read_csv('./datasets/listings.csv')
        dataframe = cleaning_dataset(dataframe)

        dataframe.to_csv('./datasets/prolog_dataframe.csv', index=False)
        cleaned_dataframe = dataframe.copy()


        print("Dataset preprocessing per il belief network")
        bn_dataset = datasetBuilding(dataframe)
        bn_dataset.to_csv('./datasets/bn_dataset.csv', index=False)
        print("Pre pocessing Belief Network eseguito.")


        dataframe = clustering_preprocessing(dataframe,cleaned_dataframe)

        dataframe.to_csv('./datasets/cleaned_dataset.csv', index=False)
        print("Pre processing eseguito.")
    except FileNotFoundError as e:
        print(e)
        print("file not found or wrong directory")

if __name__ == "__main__":
    main()
