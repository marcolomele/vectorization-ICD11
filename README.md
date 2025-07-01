# Semantic Vectorization of Hierarchical Medical Knowledge (ICD-11) via Large Language Models

A comprehensive approach to enriching and vectorizing the World Health Organization's ICD-11 medical classification system using advanced biomedical language models and embedding techniques.

## 🏥 Overview

The 11th Revision of the International Classification of Diseases (ICD-11) serves as the global diagnostic standard, organizing over 40,000 medical entities within a complex hierarchical structure. However, **ICD-11 descriptions suffer from significant information gaps** - with 7,066 entries having empty descriptions and an average completeness score of only 14% - severely hindering computational applications like semantic search and automated coding.

### Key Research Challenge
*How can we systematically enrich sparse ICD-11 descriptions and create high-quality vector representations for biomedical applications?*

**Our Solution**: Leverage Llama3-OpenBioLLM-70B to generate comprehensive medical descriptions, then evaluate multiple embedding approaches to create the first open-source vectorization of ICD-11's complex hierarchical structure.

## 📊 Dataset & Enrichment

- **13,960 ICD-11 entities** comprising 10,678 diseases (76%) and 3,282 classification entities (24%)
- **Original limitations**: 154-character average description length, 7,066 empty descriptions, 14% completeness score
- **Enhanced dataset**: 800-character average descriptions, zero empty entries, 78% completeness score

### Data Processing Pipeline
1. **WHO API Extraction** using breadth-first search to crawl the hierarchical structure
2. **Bayesian Imputation** for missing values while preserving medical relationships
3. **LLM Enhancement** via Llama3-OpenBioLLM-70B with structured prompts (causes, symptoms, transmission, diagnosis)
4. **Multi-axis Validation** across linguistic, medical, and hierarchical dimensions

## 🧠 Methodology

### Description Generation
We employ **Llama3-OpenBioLLM-70B**, chosen for its superior performance over proprietary alternatives like Med-PaLM 2 and GPT-4 on biomedical benchmarks. Our structured prompt ensures comprehensive coverage: Overview, Causes, Symptoms, Transmission, Diagnosis.

**Generation Parameters**:
- Temperature: 0.2 (deterministic reproducibility)
- Max tokens: 800 (information density)
- Structured medical format with standardized ordering

### Embedding Models Evaluation
We systematically compare **seven embedding approaches**:

#### Traditional Methods
- **TF-IDF**: 3,000 features with SVD reduction
- **FastText**: Skip-gram model, window size 5

#### Transformer-Based Models
- **BERT**: General-domain baseline (bert-base-uncased)
- **BioBERT**: PubMed abstracts fine-tuning
- **BioClinicalBERT**: MIMIC-III clinical notes specialization
- **PubMedBERT**: From-scratch biomedical training
- **GatorTron**: 82-billion token clinical corpus

### Comprehensive Evaluation Framework
Our evaluation spans multiple biomedical tasks to assess embedding quality:

1. **Intrinsic Quality**: Silhouette scores, Calinski-Harabasz indices, Davies-Bouldin scores
2. **Comorbidity Detection**: Correlation with disease co-occurrence patterns
3. **Symptom-Disease Matching**: Clinical relevance of semantic relationships
4. **Hierarchical Consistency**: Preservation of ICD-11's taxonomic structure
5. **Encyclopedia Definition Retrieval**: Accuracy in matching external medical definitions

## 🔍 Key Findings

### Description Enhancement Results
Our LLM-generated descriptions show marked improvements across all validation metrics:

| Metric | Original ICD-11 | Enhanced Descriptions |
|--------|-----------------|----------------------|
| **Completeness Score** | 14% | 78% |
| **Average Length** | 154 characters | 800 characters |
| **Empty Descriptions** | 7,066 entries | 0 entries |
| **Medical Causality** | Baseline | 6x improvement |
| **Readability (Flesch-Kincaid)** | 8.9 (intermediate) | 12-15.5 (academic) |

### Embedding Performance Rankings

#### Comorbidity Detection (Disease Co-occurrence)
| Rank | Model | Score | Interpretation |
|------|-------|-------|----------------|
| 1 | **TF-IDF** | 0.28 | Excels at lexical overlap detection |
| 2 | **GatorTron** | 0.19 | Strong clinical context understanding |
| 3 | **PubMedBERT** | 0.17 | Balanced biomedical knowledge |

#### Symptom-Disease Matching (Clinical Relevance)
| Rank | Model | Agreement Rate | Consistency |
|------|-------|----------------|-------------|
| 1 | **PubMedBERT** | 35% inter-model | High consensus |
| 2 | **BioBERT** | 35% inter-model | Medical specialization |
| 3 | **GatorTron** | 35% inter-model | Clinical notes expertise |

#### Hierarchical Structure Preservation
| Rank | Model | 1-Symbol Accuracy | 4-Symbol Accuracy |
|------|-------|-------------------|-------------------|
| 1 | **PubMedBERT** | 92.33% | 75.13% |
| 2 | **BioBERT** | 89.95% | 66.14% |
| 3 | **BERT** | 83.07% | 51.85% |

### Key Insights
- **Contextual models** (PubMedBERT, BioBERT, GatorTron) excel at semantic understanding and hierarchical relationships
- **Traditional methods** (TF-IDF, FastText) outperform in direct lexical matching tasks like comorbidity detection
- **Domain-specific training** provides crucial advantages - PubMedBERT's from-scratch biomedical training shows superior performance
- **Task-dependent performance** highlights the need for embedding selection based on specific biomedical applications

## 📂 Repository Structure
```
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

## 📈 Validation Results

### Linguistic Quality Improvements
✅ **Vocabulary Richness**: Superior Type-Token Ratio and Lexical Diversity  
✅ **Academic Readability**: Flesch-Kincaid scores 12-15.5 (vs. 8.9 original)  
✅ **Information Novelty**: Low BLEU scores (0.01-0.055) confirm substantial new content  
✅ **Medical Relevance**: 6x increase in causal medical terminology

### Medical Validation
✅ **Structured Composition**: Proper allocation of text to medical components  
✅ **Domain Alignment**: Clustering along five established medical semantic axes  
✅ **Clinical Accuracy**: Enhanced descriptions follow medical best practices  
✅ **Treatment Information**: Average 2% appropriate treatment content inclusion

### Hierarchical Consistency
✅ **Content Preservation**: Mean cosine similarity 0.636 across hierarchy levels  
✅ **Semantic Drift Management**: Expected variation in diverse subcategories  
✅ **Anatomical Distribution**: Proper emphasis on complex body systems (Head/Face, Brain)

## 🎯 Applications & Impact

### For Healthcare Professionals
**Clinical Decision Support**
- Enhanced semantic search across medical conditions
- Improved automated coding and documentation
- Better disease similarity detection for differential diagnosis

**Research Applications**
- Large-scale epidemiological studies with semantic disease grouping
- Drug discovery through disease mechanism understanding
- Clinical trial patient matching based on condition similarity

### For Technology Developers
**Healthcare AI Systems**
- Foundation for medical chatbots and virtual assistants
- Semantic search engines for medical literature
- Automated medical coding systems for hospitals

**Data Integration**
- Cross-system medical data harmonization
- Electronic health record semantic enhancement
- Medical knowledge graph construction

### For Researchers
**Biomedical NLP**
- Benchmark dataset for medical embedding evaluation
- Framework for hierarchical medical knowledge representation
- Open-source alternative to proprietary medical language models

## 🔮 Future Research Directions

### Model Architecture Improvements
- **Graph Neural Networks**: Integrate GNNs with self-supervised objectives for better hierarchical relationships
- **Foundation Model Integration**: Explore GPT-4.1, Claude Sonnet 4, and Grok 3 for enhanced embeddings
- **Scaling Law Analysis**: Investigate optimal embedding dimensions and model capacity

### Domain Adaptation
- **Multi-language Support**: Extend to non-English ICD-11 implementations
- **Temporal Dynamics**: Capture evolving medical knowledge and terminology
- **Cross-domain Transfer**: Apply methodology to other medical taxonomies (SNOMED CT, MeSH)

### Practical Applications
- **Real-time Clinical Systems**: Optimize for production deployment in hospitals
- **Patient-facing Applications**: Improve handling of informal medical language
- **Regulatory Compliance**: Ensure adherence to medical data privacy requirements

## 📚 Research Foundation

This work builds upon and extends several key research areas:

**Medical Language Models**: Leverages advances in domain-specific biomedical transformers while providing open-source alternatives to proprietary systems like Med-PaLM 2.

**Hierarchical Embeddings**: Addresses the unique challenges of ICD-11's complex, interconnected structure compared to simpler taxonomies like ICD-10.

**Biomedical Evaluation**: Introduces comprehensive evaluation frameworks specifically designed for medical embedding assessment across multiple clinical tasks.

## Contributors 🤝
* [Giorgio Caretti](https://www.linkedin.com/in/giorgio-filippo-caretti/)
* [Ilia Koldyshev](https://www.linkedin.com/in/ilia-koldyshev-b4b837287/)
* [Gleb Legotkin](https://www.linkedin.com/in/gleb-legotkin/)
* [Marco Lomele](https://www.linkedin.com/in/marco-lomele/)
* [Giovanni Mantovani](https://www.linkedin.com/in/giovanni-mantovani/)
* [Leonardo Ruzzante](https://www.linkedin.com/in/leonardo-ruzzante/)

Note: ordering is alphabetical on surname; contribution was equal across all team members. 

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@article{icd11vectorization2025,
  title={Descriptions are all you need: Semantic Vectorization of Hierarchical Medical Knowledge (ICD-11) via Large Language Models},
  author={Lomele, Marco and Legotkin, Gleb and Koldyshev, Ilia and Caretti, Giorgio and Mantovani, Giovanni and Ruzzante, Leonardo},
  institution={Bocconi University},
  year={2025}
}
```

## 🔑 License

This project is licensed under the GPL 3.0 License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Resources

- **WHO ICD-11 Official**: [https://icd.who.int/](https://icd.who.int/)
- **Llama3-OpenBioLLM**: [https://huggingface.co/aaditya/Llama3-OpenBioLLM-70B](https://huggingface.co/aaditya/Llama3-OpenBioLLM-70B)
- **Enhanced Dataset**: Available upon request for research purposes
- **Evaluation Benchmarks**: inspired from ICD2Vec https://www.sciencedirect.com/science/article/pii/S1532046423000825
