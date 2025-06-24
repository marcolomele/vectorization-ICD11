# Semantic Vectorization of Hierarchical Medical Knowledge via LLMs

This project focuses on developing a vector database and search system for the International Classification of Diseases ([ICD-11](https://icd.who.int/docs/icd-api/APIDoc-Version2/)) database. The augment ICD-11's descriptions with hierarchical information form its structure, as well as generated descriptions, for which we leverage Llama3-OpenBioLLM, an open-source model leading on bio-medical tasks. We then test 7 different embeddings modules and comprare them across a wide range of tasks, obtaining results that match previous research, in particular [ICD2Vec: Mathematical representation of diseases](https://www.sciencedirect.com/science/article/pii/S1532046423000825).

The broader context is to integrate with [Open Doctor project](https://github.com/SEBK4C/OpenDoctor-Spec) to empower translation from natural language descriptions of medical symptoms into standardized ICD codes, and then suggest relevant medical interventions, ultimately contributing to the democratization of medical knowledge while prioritizing patient privacy through local processing.

## Repository Structure 📂

```
icd-vectorization-for-open-doctor/
├── data/                 # Raw, processed, and analyzed datasets
│   ├── 1-extraction/     # Scripts for data extraction
│   ├── 2-processing/     # Notebooks for data cleaning and generation
│   └── 3-analysis/       # Notebooks for data analysis and visualizations
├── embeddings/           # Embeddings, analysis scripts, and visualizations
│   ├── embedding_analysis.py
│   ├── embeddings visuals/ # Visualizations related to embeddings
│   └── resulting ICD-11 csv embeddings/ # Stored ICD-11 embeddings
├── evaluation/           # Model evaluation notebooks and results
│   ├── comorbidity score evaluation/
│   ├── encyclopedia definition metric evaluation/
│   ├── non-medical-terms/
│   ├── symptoms benchmark/
│   └── visualizations/   # Visualizations of evaluation results
├── misc./                # Miscellaneous scripts and documentation
├── models/               # Notebooks for training various embedding models
├── report-presentation/  # Project reports and presentations
├── requirements.txt      # Python dependencies
├── LICENSE               # Project license
└── README.md             # Project overview and instructions
```

## Getting Started 🚀

To set up and run this project locally, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/vector-database-ICD.git
cd vector-database-ICD
```

### 2. Set up the environment

It is recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

Install all necessary Python packages:

```bash
pip install -r requirements.txt
```

## Data 💽

*   **data/1-extraction/**: Contains scripts like `ICDparser.py` for extracting raw ICD-11 data via a REST API. 
*   **data/2-processing/**: Includes notebooks such as `data-processing.ipynb` and `generating-descriptions.ipynb` for cleaning and enriching the data.
*   **data/3-analysis/**: Houses notebooks for in-depth data analysis and visualizations, like `data-analysis+hierarchical.ipynb`.

Processed and vectorized data, including various ICD-11 embeddings, are stored in `embeddings/resulting ICD-11 csv embeddings/`.

## Modeling 🧪

Model training and embedding generation are primarily handled within the `models/` directory. This includes notebooks for the seven different embedding approaches:

*   `berts_func_model.ipynb`: For BERT-based models (BERT, BioBERT, BioClinicalBERT, PubMedBERT).
*   `fasttext_model.ipynb`: For FastText embeddings.
*   `gatortron_model.ipynb`: For GatorTron embeddings.
*   `tfidf_model.ipynb`: For TF-IDF vectorization.

## Evaluation 📊

Comprehensive evaluation of the embedding models is conducted in the `evaluation/` directory, covering various benchmarks:

*   **Comorbidity Score Evaluation**: Notebooks and data for benchmarking using comorbidity scores.
*   **Encyclopedia Definition Metric Evaluation**: For assessing model performance against encyclopedia definitions.
*   **Non-Medical Terms Evaluation**: To analyze model robustness with non-medical prompts.
*   **Symptoms Benchmark**: For evaluating models based on symptom-based queries.

Visualizations of evaluation results are available in `evaluation/visualizations/` and `embeddings/embeddings visuals/`.

## Core Technologies 💻

This project leverages a range of technologies:

*   **Python**: Primary programming language.
*   **FastAPI**: For building the API framework.
*   **HuggingFace & Nebus AI**: For connecting to and running Llama3-OpenBioLLM.
*   **Sentence Transformers**: For generating high-quality text embeddings.
*   **ICD-11 API**: WHO's International Classification of Diseases API ([Documentation](https://icd.who.int/docs/icd-api/APIDoc-Version2/), [API Reference](https://icd.who.int/icdapi/docs2/APIDoc-Version2/)).
*   **ICHI Browser**: WHO's International Classification of Health Interventions ([Browser](https://icd.who.int/dev11/l-ichi/en)).

## External Resources 🔗

*   [ICD API Homepage](https://icd.who.int/icdapi)
*   [ICD API Documentation v2.x](https://icd.who.int/docs/icd-api/APIDoc-Version2/)
*   [ICD API Reference (Swagger)](https://icd.who.int/icdapi/docs2/APIDoc-Version2/)
*   [Supported Classification Versions](https://icd.who.int/icdapi/docs2/SupportedClassifications/)
*   [Clinical Table Search Service API for ICD-11](https://clinicaltables.nlm.nih.gov/apidoc/icd11_codes/v3/doc.html)
*   [NLTK Documentation](https://www.nltk.org/)
*   [scikit-learn Documentation](https://scikit-learn.org/)

## License 🔑

This project is licensed under the terms of the MIT License. See the `LICENSE` file for more details.
