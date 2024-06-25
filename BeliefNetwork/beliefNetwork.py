import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator

class BeliefNetwork:
    model = None
    inference_method = None
    beliefNet_structure ='./datasets/bnStructure.xml'
    data_path = './datasets/bn_dataset.csv'

    def __init__(self):
        """
        Costruttore della classe BeliefNetwork.
        """
        self.model = self.train_model()
        self.inference_method = VariableElimination(self.model)

    def inference(self, preferences_dictionary):
        """
        Esegue un'infrazione sulla rete bayesiana e restituisce i risultati.

        :param preferences_dictionary: Dizionario contenente le preferenze come evidenza per l'infrazione.
        :return: Dizionario contenente i risultati dell'infrazione.
        """
        results_dictionary = {}
        result = self.inference_method.query(variables=['review_scores_rating'], evidence=preferences_dictionary)
        results_dictionary['top_rating'] = result.get_value(review_scores_rating='top_rating').round(4)
        results_dictionary['nice_rating'] = result.get_value(review_scores_rating='nice_rating').round(4)
        results_dictionary['good_rating'] = result.get_value(review_scores_rating='good_rating').round(4)
        results_dictionary['low_rating'] = result.get_value(review_scores_rating='low_rating').round(4)
        return results_dictionary

    def compute_query(self, input_string):
        """
        Elabora una stringa di input e restituisce un dizionario di preferenze.

        :param input_string: Stringa contenente le preferenze separate da virgola nel formato 'chiave=valore'.
        :return: Dizionario contenente le preferenze elaborate dalla stringa di input.
        """
        input_string = input_string
        preferences_list = input_string.split(",")
        preferences_dictionary = {}
        for item in preferences_list:
            key = (item.split("=")[0])
            value = str(item.split("=")[1])
            preferences_dictionary[key] = value
        return preferences_dictionary

    def train_model(self):
        # Caricamento del dataset
        df = pd.read_csv('./datasets/bn_dataset.csv')
        
        # Creazione del dizionario
        variables = {col: df[col].unique().tolist() for col in df.columns}
        
        # Definizione delle relazioni tra le variabili
        relationships = [
            ("class_of_price", "neighbourhood_cleansed"),
            ("neighbourhood_cleansed", "is_center"),
            ("is_center", "number_of_reviews"),
            ("number_of_reviews", "review_scores_rating"),
            ("review_scores_rating", "host_response_rate"),
            ("review_scores_rating", "host_is_superhost"),
            ("host_response_rate", "long_term_stays_allowed"),
            ("hot_water_kettle", "smoke_alarm"),
            ("hair_dryer", "iron"),
            ("review_scores_rating", "hangers"),
            ("review_scores_rating", "cooking_basics"),
            ("hair_dryer", "shampoo"),
            ("review_scores_rating", "hair_dryer"),
            ("review_scores_rating", "essentials"),
            ("review_scores_rating", "heating"),
            ("washer", "kitchen"),
            ("washer", "wifi"),
            ("review_scores_rating", "washer"),
            ("review_scores_rating", "coffee_maker"),
            ("review_scores_rating", "hot_water"),
            ("review_scores_rating", "dishes_and_silverware"),
            ("review_scores_rating", "oven"),
            ("review_scores_rating", "dishwasher"),
            ("review_scores_rating", "refrigerator"),
            ("review_scores_rating", "bed_linens"),
            ("review_scores_rating", "microwave"),
            ("review_scores_rating", "stove"),
            ("review_scores_rating", "tv"),
            ("review_scores_rating", "dedicated_workspace"),
            ("review_scores_rating", "cleaning_products"),
            ("bed_linens", "hot_water_kettle"),
            ("hot_water_kettle", "dining_table")
        ]

        # Inizializzazione del modello BayesianNetwork
        model_temp = BayesianNetwork(relationships)
        
        # Addestramento del modello per calcolare le CPD
        model_temp.fit(df, estimator=MaximumLikelihoodEstimator)
        
        # Costruzione delle CPDs
        cpds = []
        for cpd_temp in model_temp.get_cpds():
            # Verifica se la CPD ha evidence
            has_evidence = len(cpd_temp.get_evidence()) > 0
            
            # Definire le variabili di interesse
            variables_of_interest = [cpd_temp.variable]
            if has_evidence:
                variables_of_interest.append(cpd_temp.get_evidence()[0])
            
            # Estrazione delle informazioni relative alle variabili di interesse dal dizionario variables
            variables_subset = {var: variables[var] for var in variables_of_interest}
            
            # Costruzione degli state_names dal dizionario variables_subset
            state_names = {var: [str(state) for state in states] for var, states in variables_subset.items()}
            
            # Costruzione delle CPDs
            if has_evidence:
                cpd = TabularCPD(variable=cpd_temp.variable, 
                                variable_card=len(variables_subset[cpd_temp.variable]),
                                values=cpd_temp.values,
                                evidence=cpd_temp.get_evidence(),
                                evidence_card=[len(variables_subset[cpd_temp.get_evidence()[0]])],
                                state_names=state_names)
            else:
                cpd = TabularCPD(variable=cpd_temp.variable, 
                                variable_card=len(variables_subset[cpd_temp.variable]),
                                values=[[v] for v in cpd_temp.values],
                                state_names=state_names)
            cpds.append(cpd)
        
        # Crea un nuovo oggetto BayesianNetwork
        model = BayesianNetwork()
        # Aggiungi le relazioni tra i nodi
        model.add_edges_from(relationships)
        # Aggiungi i nodi alla rete basati sui CPD
        model.add_cpds(*cpds)

        # Verifica il modello
        model.check_model()

        # Restituisci il modello addestrato
        return model
