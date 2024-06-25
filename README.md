# ICON 2023/2024
### Fracchiolla - Grandolfo
Repository per il progetto realizzato per l'esame di Ingegneria della Conoscenza del dipartimento di Informatica dell'Università degli studi di Bari Aldo Moro.

> Usare *'Git Bash'* su Windows

## Configurazione del progetto
- Installare SWI-Prolog sulla propria macchina <br />
https://www.swi-prolog.org/download/stable/bin/swipl-8.2.4-1.x64.exe.envelope

- Clonare il progetto
```
git clone https://github.com/dgrandolfo4/ICON2324.git
```

- Posizionarsi nella repository clonata
```
cd ICON2324
```

- Creare il virtual env
```python
python -m venv venv
```

- Attivare il virtual env
```python
source venv/Scripts/activate
```

- Installare le dipendenze
```python
pip install -r requirements.txt
```

## Esecuzione
### Esecuzione automatica delle fasi di preprocessing
```python
python preprocessing/preprocessing.py
```

### Esecuzione manuale delle fasi di preprocessing
- Eseguire la fase di preprocessing
```python
python preprocessing/cleaning.py
```

- Creare i clusters
```python
python clustering/clustering.py <number of iterations (optional)>
```

- Creare la Knowledge Base
```python
python KnowledgeBase/Kb.py
```

- UI per la predisposizione di queries
```python
python BeliefNetwork/ui.py
```