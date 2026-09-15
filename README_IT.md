# Credit Risk Model Development Platform

**Lingua:** [English](README.md) | [Deutsch](README_DE.md) | [Italiano](README_IT.md)

Un progetto end-to-end di machine learning per la modellazione del rischio di insolvenza creditizia, con acquisizione riproducibile dei dati, validazione, preprocessing, confronto dei modelli, calibrazione delle probabilità, analisi delle soglie decisionali, spiegabilità, tracciamento degli esperimenti, reporting e test automatizzati.

> **Progetto portfolio:** Questo repository dimostra concetti di sviluppo di modelli di rischio di credito e pratiche di software engineering. Non dichiara conformità normativa, conformità IRBA o idoneità all'uso in produzione.

## Panoramica del progetto

Il progetto utilizza il dataset UCI **Default of Credit Card Clients** per sviluppare e valutare modelli destinati a prevedere l'insolvenza di pagamento nel mese successivo.

Il workflow attuale è:

```text
Dataset UCI
    |
    v
Acquisizione dei dati
    |
    v
Validazione dei dati
    |
    v
Preprocessing
    |
    v
Suddivisione Training / Holdout
    |
    v
Confronto dei modelli con Cross-Validation
    |
    v
Calibrazione delle probabilità
    |
    +--> Analisi delle soglie
    |
    +--> Spiegabilità
    |
    v
MLflow Experiment Tracking
    |
    v
Reporting e Testing automatizzati
```

## Funzionalità implementate

- Acquisizione riproducibile del dataset UCI
- Validazione dello schema e della qualità dei dati
- Pipeline di preprocessing per i dati di rischio di credito
- Suddivisione stratificata in dati di training e holdout
- Logistic Regression come baseline
- Decision Tree
- Random Forest
- Cross-validation stratificata a 5 fold
- Valutazione con ROC-AUC, Average Precision, Brier Score e Log Loss
- Analisi delle soglie decisionali
- Calibrazione delle probabilità con metodi sigmoid e isotonic
- Spiegabilità del modello mediante Permutation Importance
- Visualizzazioni ROC, Precision-Recall, calibrazione, soglie e Feature Importance
- Report automatizzato del modello in formato Markdown
- MLflow Experiment Tracking con backend SQLite locale
- Serializzazione del modello MLflow mediante `skops`
- Suite automatizzata di 54 test
- Controlli statici della qualità del codice con Ruff
- Controllo di versione con Git e GitHub

## Sviluppo dei modelli

Vengono confrontati tre modelli candidati:

| Modello | ROC-AUC media CV | Average Precision media CV |
| --- | ---: | ---: |
| Random Forest | 0.7674 | 0.5397 |
| Logistic Regression | 0.7272 | 0.5068 |
| Decision Tree | 0.6149 | 0.2908 |

Tra i modelli candidati valutati, il Random Forest ha ottenuto la migliore capacità discriminante in cross-validation.

### Calibrazione delle probabilità

Il Random Forest è stato successivamente valutato con calibrazione sigmoid e isotonic.

La selezione del metodo di calibrazione è stata effettuata tramite cross-validation sui soli dati di training, utilizzando il **Brier Score** come criterio primario di calibrazione.

| Metodo | ROC-AUC media CV | Brier Score medio CV | Log Loss medio CV |
| --- | ---: | ---: | ---: |
| Isotonic | 0.7736 | **0.1355** | **0.4321** |
| Sigmoid | **0.7741** | 0.1357 | 0.4331 |
| Non calibrato | 0.7674 | 0.1376 | 0.4405 |

La calibrazione isotonic è stata selezionata perché ha ottenuto il Brier Score medio più basso in cross-validation.

## Valutazione del modello selezionato

Il Random Forest selezionato con calibrazione isotonic ha prodotto:

| Metrica | Risultato Holdout |
| --- | ---: |
| ROC-AUC | 0.7625 |
| Average Precision | 0.5425 |
| Brier Score | 0.1375 |
| Log Loss | 0.4381 |

I risultati indicano una capacità discriminante predittiva moderata e un miglioramento della qualità delle probabilità rispetto al Random Forest non calibrato.

**Nota metodologica:** il metodo di calibrazione è stato selezionato mediante cross-validation sui dati di training. Il dataset holdout riportato qui era già stato esaminato durante fasi precedenti dello sviluppo del modello e non deve quindi essere interpretato come un nuovo test set finale e indipendente.

## Analisi delle soglie

Il progetto valuta diverse soglie di probabilità anziché assumere automaticamente che `0.5` sia la soglia appropriata.

Questo permette di analizzare il compromesso tra:

- False Negatives: casi di insolvenza non identificati dal modello come rischiosi
- False Positives: casi senza insolvenza identificati come rischiosi
- Recall
- Precision
- False-Positive Rate

Non viene dichiarata alcuna soglia ottimale dal punto di vista economico, poiché il dataset UCI non fornisce le ipotesi sui costi necessarie per tale decisione.

## Spiegabilità

La Permutation Importance viene utilizzata per analizzare quanto il Random Forest addestrato dipenda dalle singole variabili di input.

Tra le dipendenze più forti osservate nel modello figurano:

1. `PAY_0`
2. `LIMIT_BAL`
3. `PAY_2`
4. `BILL_AMT1`
5. `PAY_3`

La Permutation Importance misura la **dipendenza del modello, non la causalità**. Predittori correlati possono inoltre distribuire o ridurre l'importanza misurata.

## Experiment Tracking

MLflow viene utilizzato per tracciare l'esperimento relativo al modello selezionato.

La configurazione locale attuale registra:

- parametri del modello
- ROC-AUC
- Average Precision
- Brier Score
- Log Loss
- artefatto serializzato del modello scikit-learn
- metadati MLflow relativi all'ambiente e al modello

I metadati di tracking vengono memorizzati tramite un backend SQLite locale. I database MLflow locali e gli artefatti di tracking generati sono esclusi dal controllo di versione Git.

## Dataset

Il progetto utilizza il dataset **Default of Credit Card Clients** dell'UCI Machine Learning Repository.

Il dataset contiene:

- 30.000 osservazioni
- un target binario di insolvenza
- informazioni sul limite di credito
- storico dello stato dei pagamenti
- importi degli estratti conto
- importi dei pagamenti precedenti
- caratteristiche demografiche

Fonte del dataset:

https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

DOI:

https://doi.org/10.24432/C55S3H

Il dataset è distribuito con licenza **CC BY 4.0**.

### Limitazioni del dataset

Il dataset è storico e non rappresenta gli attuali portafogli bancari europei.

Non contiene tutte le informazioni necessarie per la gestione del rischio di credito in produzione o per lo sviluppo di modelli regolamentari.

Questo progetto pertanto **non dichiara**:

- conformità IRBA
- validazione regolamentare del modello
- idoneità alla produzione
- rappresentatività degli attuali portafogli di credito europei

## Stack tecnologico

### Implementato

- Python 3.12
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- MLflow
- SQLite
- pytest
- Ruff
- Git
- GitHub

### Estensioni pianificate della piattaforma

I seguenti componenti sono estensioni pianificate e **non vengono ancora presentati come funzionalità implementate**:

- PostgreSQL per la persistenza dei dati della piattaforma
- Apache Airflow per l'orchestrazione dei workflow
- Docker / Docker Compose per servizi riproducibili
- GitHub Actions per la Continuous Integration

## Struttura del progetto

```text
credit-risk-platform/
|-- config/
|-- data/
|   |-- raw/
|   `-- processed/
|-- docker/
|-- notebooks/
|-- reports/
|   `-- figures/
|-- src/
|   `-- credit_risk_platform/
|-- tests/
|-- pyproject.toml
|-- .gitignore
`-- README.md
```

## Test e qualità del codice

La suite automatizzata attuale comprende **54 test** che coprono componenti fondamentali, tra cui:

- preprocessing
- modellazione
- valutazione
- analisi delle soglie
- calibrazione
- spiegabilità
- visualizzazione
- reporting
- MLflow Experiment Tracking

Per l'attuale milestone verificata del progetto:

```text
54 passed
Ruff: All checks passed
```

Questi risultati corrispondono al quality gate locale verificato per l'attuale milestone del progetto.

## Riproducibilità

Le dipendenze del progetto sono definite in `pyproject.toml`.

Il progetto attualmente richiede:

```text
Python >=3.12,<3.13
```

Il workflow di sviluppo locale utilizza un ambiente virtuale Python isolato.

Un ambiente completamente containerizzato non è ancora stato implementato. La riproducibilità basata su Docker fa parte delle estensioni pianificate della piattaforma.

## Roadmap

- [x] Acquisizione dei dati
- [x] Validazione dei dati
- [x] Preprocessing
- [x] Modelli baseline e candidati
- [x] Cross-validation
- [x] Valutazione holdout
- [x] Analisi delle soglie
- [x] Spiegabilità
- [x] Calibrazione delle probabilità
- [x] Reporting automatizzato
- [x] MLflow Experiment Tracking
- [x] Test automatizzati
- [ ] Persistenza PostgreSQL
- [ ] Orchestrazione Apache Airflow
- [ ] Ambiente Docker Compose
- [ ] GitHub Actions CI

## Disclaimer

Questo repository è un'implementazione didattica e di portfolio ispirata ai workflow professionali per lo sviluppo di modelli di rischio di credito.

I modelli, le procedure di validazione, i dati e l'architettura software presentati non sono sufficienti a dimostrare conformità normativa, conformità IRBA o idoneità all'uso in produzione.