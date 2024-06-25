import pandas as pd
import pyswip as psw
import re

try:
    from beliefNetwork import BeliefNetwork
except ModuleNotFoundError:
    from .beliefNetwork import BeliefNetwork
prg = psw.Prolog() #oggetto prolog per eseguire query

def BF_help():
    """
    print dell'help della Belief Network
    """
    df = pd.read_csv('./datasets/bn_dataset.csv')
    # Identifica le colonne con valori booleani
    boolean_columns = [col for col in df.columns if set(df[col].dropna().unique()).issubset({True, False})]
    # Ottieni le colonne rimanenti
    discrete_columns = [col for col in df.columns if col not in boolean_columns]

    print("\nAttributi discrete disponibili:")
    for col in discrete_columns:
        unique_values = df[col].unique()
        unique_values = set(unique_values)
        unique_values_str = ", ".join(map(str, unique_values))
        print(f"- {col.title().replace("_", " ")} --> Valori {{{unique_values_str}}}")
    print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    print("\nAttributi boolean disponibili:")
    for col in sorted(boolean_columns):
        # Stampa il nome della colonna
        print(f"- {col}")
    print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    print("\nInserisci le evidenze per la  belief network rispettando il seguente formato:")
    print("NomeAttributo = valore, NomeAttributo = valore, ...\n"
        "I nomi degli attributi devono essere scritti in minuscolo.")
    print("\nEsempio: host_is_superhost = True, class_of_price = economy")
    print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------")



def correct_input(options):
    """
    Richiede un input all'utente e verifica se è tra le opzioni valide.
    """
    while True:
        command = input().lower().strip()
        if command in options:
            return command
        print("Comando sbagliato, puoi inserire: ", options)

def dropping(answer):
    """
    Elimina le righe del DataFrame in base alle variabili Prolog non istanziate.
    """
    dataframe = pd.DataFrame(answer)
    canc = {index for index, row in dataframe.iterrows() if any(isinstance(row[col], psw.Variable) for col in dataframe.columns)}
    dataframe.drop(index=canc, inplace=True)
    return dataframe

def bn_inference():
    """
    Gestisce le query per la Belief Network.
    """
    b = BeliefNetwork()
    BF_help()
    while True:
        print("\nInserire le tue preferenze:")
        preferences = input().replace(' ', '')
        if re.match('((([a-z]+)([_]([a-z]+))*)([=])(([a-z|A-Z]+)([_]([a-z]+))*)([,]*))+', preferences):
            try:
                results = b.inference(b.compute_query(preferences))
                print("{:<15} {:<15}".format('RATING', 'PROBABILITY'))
                for key, value in results.items():
                    print("{:<15} {:<15}".format(key, value))
            except Exception as e:
                print("Error:", e)

            print("\nVuoi inserire un'altra query? [si, no]")
            response = correct_input(['si', 'no'])
            if response == 'no':
                break
        else:
            print("Formato non corretto. RIPETERE!")

def kb_query():
    """
    Gestisce le query per la Knowledge Base.
    """
    try:
        print("\nInserire una query per la Knowledge Base:")
        query = input()
        # Correggere eventuali virgolette curve
        query = query.replace('“', '"').replace('”', '"')
        # Esegue la query Prolog
        answer = prg.query(query)
        answer_list = list(answer)
        if not answer_list:
            print("false")
        elif len(answer_list) == 1 and not answer_list[0]:
            print("true")
        else:
            dataframe = dropping(answer_list)
            print("OUTPUT:\n")
            for col in dataframe.columns:
                print("Index\t" + col)
                i = 0
                for row in dataframe[col].explode().to_string(index=False).split("\n"):
                    print(str(i) + "\t" + row)
                    i += 1
    except Exception as e:
        print("Error:", e)

def menu():
    """
    Stampa un menu di aiuto per l'utente.
    """
    print("\nMENU'\n")
    print("'query' -> per porre una query alla Knowledge Base")
    print("'inference' -> per eseguire un'inference con Belief Network")
    print("'esci' -> per uscire\n")

def main():
    """
    Funzione principale che coordina l'esecuzione del programma.
    """
    print("\nCaricamento knowledge base...", end="", flush=True)
    prg.consult("./datasets/kb.pl")
    pd.set_option('display.max_rows', 3000, 'display.max_columns', 10)
    while True:
        menu()
        print("Inserire:")
        command = input().strip().lower()
        if command == 'esci':
            break
        elif command == 'query':
            while True:
                kb_query()
                print("\nVuoi inserire un'altra query? [si, no]")
                cont = correct_input(['si', 'no'])
                if cont == 'no':
                    break
        elif command == 'inference':
            bn_inference()
        else:
            print("comando ERRATO. RIPETERE!")
            menu()

if __name__ == "__main__":
    main()