# 💳 Credit Risk Model Development Platform

**Modellazione end-to-end del rischio di default creditizio — dai dati grezzi alle probabilità calibrate, all'explainability, al monitoraggio degli esperimenti e alla reportistica automatizzata.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Code Quality](https://img.shields.io/badge/Ruff-passing-brightgreen)](https://docs.astral.sh/ruff/)
[![CI](https://github.com/Umi156/credit-risk-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/Umi156/credit-risk-platform/actions/workflows/ci.yml)

**Lingua:** 🇬🇧 [English](README.md) · 🇩🇪 [Deutsch](README_DE.md) · 🇮🇹 [Italiano](README_IT.md)

---

## 🎯 Il progetto in sintesi

Questo progetto di portfolio implementa un workflow di machine learning riproducibile per la **modellazione del rischio di default creditizio**, utilizzando il dataset UCI *Default of Credit Card Clients*.

Il progetto va oltre il semplice addestramento del modello e comprende **validazione dei dati, cross-validation, calibrazione delle probabilità, analisi delle soglie decisionali, explainability, experiment tracking con MLflow, reportistica automatizzata, test del software, persistenza PostgreSQL e Continuous Integration**.

> [!IMPORTANT]
> **Ambito del portfolio:** questo progetto dimostra concetti di sviluppo di modelli di rischio di credito e pratiche di software engineering. **Non dichiara conformità normativa, conformità IRBA o idoneità alla produzione.**

### Cosa dimostra questo progetto

| Area | Implementazione |
| --- | --- |
| 💳 **Rischio di credito** | Modellazione del rischio / della probabilità di default |
| 🧠 **Machine Learning** | Logistic Regression, Decision Tree, Random Forest |
| 📊 **Validazione del modello** | Stratified 5-fold CV, ROC-AUC, AP, Brier Score, Log Loss |
| 🎯 **Calibrazione** | Calibrazione sigmoid e isotonic delle probabilità |
| ⚖️ **Analisi decisionale** | Trade-off delle soglie, precision, recall e false-positive rate |
| 🔍 **Explainability** | Permutation Feature Importance |
| 🧪 **Experiment Tracking** | MLflow con backend SQLite locale |
| 🗄️ **Persistenza dati** | PostgreSQL tramite `psycopg`, configurazione via ambiente, caricamento bulk con `COPY` |
| 🛠️ **Engineering** | Struttura package Python, pytest, Ruff, Git/GitHub, GitHub Actions |
| 📄 **Reporting** | Report automatizzato del modello e visualizzazioni diagnostiche |

---

## 🏆 Risultati in sintesi

### Modello selezionato — Random Forest con calibrazione Isotonic

| ROC-AUC ↑ | Average Precision ↑ | Brier Score ↓ | Log Loss ↓ |
| :---: | :---: | :---: | :---: |
| **0.7625** | **0.5425** | **0.1375** | **0.4381** |

Il metodo di calibrazione è stato selezionato mediante **cross-validation sui dati di training**, utilizzando il Brier Score come criterio principale di calibrazione.

> [!NOTE]
> Il dataset holdout era già stato analizzato durante precedenti fasi di sviluppo. Questi valori rappresentano quindi la valutazione corrente del progetto di portfolio e **non un nuovo test set finale completamente indipendente**.

---

## 🔄 Workflow end-to-end

```text
                    UCI Credit Card Default Dataset
                                 │
                                 ▼
                         Data Ingestion
                                 │
                                 ▼
                         Data Validation
                                 │
                                 ▼
                    PostgreSQL Persistence
                                 │
                                 ▼
                          Preprocessing
                                 │
                                 ▼
                     Stratified Train / Holdout
                                 │
                                 ▼
                 5-Fold Cross-Validated Comparison
                        ┌────────┼────────┐
                        ▼        ▼        ▼
                     Logistic  Decision  Random
                    Regression   Tree    Forest
                                           │
                                           ▼
                                  Probability Calibration
                                    ┌──────┴──────┐
                                    ▼             ▼
                                Sigmoid       Isotonic
                                                   │
                                                   ▼
                                          Selected Model
                                                   │
                         ┌─────────────────────────┼─────────────────────┐
                         ▼                         ▼                     ▼
                 Threshold Analysis        Explainability        MLflow Tracking
                         │                         │                     │
                         └─────────────────────────┼─────────────────────┘
                                                   ▼
                                      Reporting & Automated Tests
                                                   │
                                                   ▼
                                      GitHub Actions CI
```

---

## 🛠️ Funzionalità implementate

- Acquisizione riproducibile del dataset sorgente UCI
- Validazione dello schema e della qualità dei dati
- Persistenza PostgreSQL dei dati sorgente validati tramite `psycopg`
- Caricamento bulk con PostgreSQL `COPY` e vincoli database
- Integrità del round trip Pandas → PostgreSQL → Pandas verificata
- Pipeline di preprocessing per i dati di rischio di credito
- Suddivisione stratificata in training e holdout
- Baseline con Logistic Regression
- Modello Decision Tree
- Modello Random Forest
- Cross-validation stratificata a 5 fold
- Valutazione con ROC-AUC, Average Precision, Brier Score e Log Loss
- Analisi delle soglie decisionali
- Calibrazione delle probabilità con sigmoid e isotonic calibration
- Explainability del modello mediante Permutation Importance
- Visualizzazioni ROC, Precision-Recall, calibration, threshold e feature importance
- Report automatizzato del modello in Markdown
- MLflow Experiment Tracking con backend SQLite locale
- Serializzazione del modello MLflow tramite `skops`
- Suite automatizzata
- Controlli statici della qualità del codice con Ruff
- Controllo di versione con Git/GitHub
- Continuous Integration con GitHub Actions

---

## 🧠 Sviluppo del modello

Vengono confrontati tre modelli candidati:

| Modello | Mean CV ROC-AUC | Mean CV Average Precision |
| --- | ---: | ---: |
| **Random Forest** | **0.7674** | **0.5397** |
| Logistic Regression | 0.7272 | 0.5068 |
| Decision Tree | 0.6149 | 0.2908 |

Il Random Forest ha ottenuto la migliore capacità discriminante in cross-validation tra i modelli candidati valutati.

### Calibrazione delle probabilità

Il Random Forest è stato successivamente valutato con calibrazione sigmoid e isotonic.

La selezione del metodo di calibrazione è stata effettuata mediante cross-validation sui dati di training, utilizzando il **Brier Score** come criterio principale di calibrazione.

| Metodo | Mean CV ROC-AUC | Mean CV Brier Score | Mean CV Log Loss |
| --- | ---: | ---: | ---: |
| **Isotonic** | 0.7736 | **0.1355** | **0.4321** |
| Sigmoid | **0.7741** | 0.1357 | 0.4331 |
| Uncalibrated | 0.7674 | 0.1376 | 0.4405 |

La calibrazione isotonic è stata selezionata perché ha ottenuto il Brier Score medio più basso nella cross-validation.

---

## 📊 Valutazione del modello selezionato

Il Random Forest selezionato con calibrazione isotonic ha prodotto:

| Metrica | Risultato holdout | Direzione |
| --- | ---: | :---: |
| ROC-AUC | **0.7625** | ↑ più alto è meglio |
| Average Precision | **0.5425** | ↑ più alto è meglio |
| Brier Score | **0.1375** | ↓ più basso è meglio |
| Log Loss | **0.4381** | ↓ più basso è meglio |

I risultati indicano una capacità discriminante moderata e un miglioramento della qualità delle probabilità rispetto al Random Forest non calibrato.

> [!CAUTION]
> **Limitazione metodologica:** il metodo di calibrazione è stato selezionato mediante cross-validation sui dati di training. Il dataset holdout riportato qui era già stato analizzato durante precedenti fasi di sviluppo e non deve quindi essere interpretato come un nuovo test set finale indipendente.

---

## 📈 Diagnostica del modello

### Curve ROC

Le curve ROC confrontano la capacità di ranking e discriminazione dei modelli candidati valutati.

![ROC curves](reports/figures/roc_curves.png)

### Calibrazione delle probabilità

L'analisi di calibrazione confronta le probabilità di default previste con le frequenze di default osservate.

![Calibration method comparison](reports/figures/calibration_method_comparison.png)

### Explainability del modello

La Permutation Importance viene utilizzata per analizzare quanto il Random Forest addestrato dipenda dalle singole feature di input.

![Permutation feature importance](reports/figures/permutation_importance.png)

---

## ⚖️ Analisi delle soglie

Il progetto valuta diverse soglie di probabilità anziché assumere automaticamente che `0.5` sia appropriato.

Questo permette di analizzare i trade-off tra:

- **False negatives** — casi a rischio di default non identificati dal modello
- **False positives** — casi senza default classificati come rischiosi
- **Recall**
- **Precision**
- **False-positive rate**

Non viene dichiarata una soglia ottimale dal punto di vista economico, poiché il dataset UCI non contiene le ipotesi sui costi necessarie per prendere tale decisione.

---

## 🔍 Explainability

La Permutation Importance viene utilizzata per analizzare quanto il Random Forest addestrato dipenda dalle singole feature di input.

Le dipendenze dal modello più forti osservate includono:

| Posizione | Feature |
| ---: | --- |
| 1 | `PAY_0` |
| 2 | `LIMIT_BAL` |
| 3 | `PAY_2` |
| 4 | `BILL_AMT1` |
| 5 | `PAY_3` |

> [!NOTE]
> La Permutation Importance misura la **dipendenza del modello, non la causalità**. Predictor correlati possono inoltre distribuire o ridurre l'importance misurata tra più feature.

---

## 🧪 Experiment Tracking

MLflow viene utilizzato per il tracking dell'esperimento relativo al modello selezionato.

La configurazione locale corrente registra:

- parametri del modello
- ROC-AUC
- Average Precision
- Brier Score
- Log Loss
- artefatto serializzato del modello scikit-learn
- metadati MLflow relativi all'ambiente e al modello

I metadati di tracking vengono archiviati tramite un **backend SQLite locale**.

I database MLflow locali e gli artefatti di tracking generati sono esclusi dal controllo di versione Git.

---

## 🗄️ Persistenza PostgreSQL

I dati sorgente validati possono essere persistiti nella tabella PostgreSQL `validated_credit_data`. Le impostazioni di connessione vengono lette da variabili d'ambiente, evitando di archiviare le credenziali del database nel repository.

Il layer di persistenza utilizza PostgreSQL `COPY` per il caricamento bulk e applica una primary key, vincoli `NOT NULL` e un check constraint binario per `default_flag`. La verifica locale di integrazione ha confermato **30.000 righe**, **30.000 ID univoci** e **6.636 default osservati**. È stato inoltre verificato un round trip Pandas → PostgreSQL → Pandas con shape, colonne e valori dei dati identici.

L'integrazione PostgreSQL è attualmente verificata in locale. Il workflow GitHub Actions rimane indipendente dal database e utilizza unit test del database con mock.

---

## 📦 Dataset

Il progetto utilizza il dataset **UCI Machine Learning Repository — Default of Credit Card Clients**.

| Proprietà | Valore |
| --- | --- |
| Osservazioni | 30.000 |
| Target | Indicatore binario di default |
| Informazioni sul credito | Limite di credito |
| Comportamento di pagamento | Storico dello stato dei pagamenti |
| Storico finanziario | Importi fatturati e pagamenti precedenti |
| Altri attributi | Variabili demografiche |

**Fonte del dataset:**

https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

**DOI:**

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

---

## ⚙️ Stack tecnologico

### Implementato

| Categoria | Tecnologie |
| --- | --- |
| Linguaggio | Python 3.12 |
| Dati | Pandas, NumPy |
| Machine Learning | scikit-learn |
| Visualizzazione | Matplotlib, Seaborn |
| Experiment Tracking | MLflow |
| Backend di tracking | SQLite |
| Persistenza dati | PostgreSQL tramite `psycopg` |
| Testing | pytest |
| Qualità del codice | Ruff |
| Controllo di versione | Git, GitHub |
| Continuous Integration | GitHub Actions |

### Estensioni della piattaforma pianificate

I seguenti componenti sono estensioni pianificate e **non rappresentano ancora funzionalità completate**:

- Apache Airflow per l'orchestrazione dei workflow
- Docker / Docker Compose per servizi riproducibili

---

## 📁 Struttura del progetto

```text
credit-risk-platform/
├── .github/
│   └── workflows/
│       └── ci.yml
├── config/
├── data/
│   ├── raw/
│   └── processed/
├── docker/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   └── credit_risk_platform/
│       ├── database.py
│       └── persistence.py
├── tests/
├── README.md
├── README_DE.md
├── README_IT.md
├── pyproject.toml
└── .gitignore
```

---

## ✅ Test e qualità del codice

La suite di test automatizzata copre componenti fondamentali tra cui:

- preprocessing
- modeling
- evaluation
- threshold analysis
- calibration
- explainability
- visualization
- reporting
- MLflow experiment tracking
- connessione e persistenza PostgreSQL

Quality gate attualmente verificati:

```text
Locale:
61 passed
Ruff: All checks passed

GitHub Actions CI:
success
```

Il workflow GitHub Actions riproduce il quality gate del progetto su un nuovo runner Ubuntu con Python 3.12, installando le dipendenze del progetto, recuperando il dataset sorgente UCI ed eseguendo Ruff e la suite automatizzata pytest.

---

## 🔁 Riproducibilità

Le dipendenze del progetto sono definite in `pyproject.toml`.

Il progetto attualmente utilizza:

```text
Python >=3.12,<3.13
```

Il workflow di sviluppo locale utilizza un ambiente virtuale Python isolato.

La Continuous Integration viene eseguita su un nuovo runner Ubuntu di GitHub Actions con Python 3.12.

Un ambiente completamente containerizzato non è ancora stato implementato. La riproducibilità basata su Docker rimane parte delle estensioni pianificate della piattaforma.

---

## 🗺️ Roadmap

- [x] Data ingestion
- [x] Data validation
- [x] Preprocessing
- [x] Modelli baseline e candidati
- [x] Cross-validation
- [x] Valutazione holdout
- [x] Threshold analysis
- [x] Explainability
- [x] Probability calibration
- [x] Reporting automatizzato
- [x] MLflow Experiment Tracking
- [x] Test automatizzati
- [x] GitHub Actions CI
- [x] Persistenza PostgreSQL
- [ ] Orchestrazione con Apache Airflow
- [ ] Ambiente Docker Compose

---

## ⚠️ Disclaimer

Questo repository è un'implementazione educativa e di portfolio ispirata ai workflow professionali di sviluppo dei modelli di rischio di credito.

I modelli, le procedure di validazione, il dataset e l'architettura software presentati qui **non sono sufficienti per dimostrare conformità normativa, conformità IRBA o idoneità alla produzione**.