import sys
import os
import cleaning
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from clustering import clustering as cl
from KnowledgeBase import Kb as kb
from BeliefNetwork import ui

def main():
    try:
        print("Esecuzione automatica delle fasi di preprocessing...\n\n")

        # Eseguire il main di cleaning
        cleaning.main()
        print("Cleaning completato.\n")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

    try:
        # Eseguire il main di clustering
        cl.main()
        print("Clustering completato.\n")

        # Eseguire il main di KnowledgeBase
        kb.main()
        print("KnowledgeBase completato.\n")

        # Richiesta di conferma per l'esecuzione di ui.main
        while True:
            confirm = input("Vuoi eseguire l'interfaccia utente per la predisposizione di query? (y/n): ").strip().lower()
            if confirm == 'y':
                ui.main()
                break
            elif confirm == 'n':
                print("L'esecuzione dell'interfaccia utente è stata annullata.")
                print("Puoi sempre avviarla tramite il comando 'python BeliefNetwork/ui.py'")
                break
            else:
                print("Risposta non valida. Per favore, rispondi con 'y' (sì) o 'n' (no).")

    except Exception as e:
        print(f"Si è verificato un errore: {e}")

if __name__ == "__main__":
    main()
