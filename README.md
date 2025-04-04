# ICD Vectorization for Open Doctor - NLP Student Project

## Project Overview

This project aims to develop a vector database and search system for the International Classification of Diseases ([ICD-11](https://icd.who.int/docs/icd-api/APIDoc-Version2/)) database, enabling natural language querying for medical conditions and appropriate interventions. This work is part of the larger [Open Doctor project](https://github.com/SEBK4C/OpenDoctor-Spec), which provides AI-powered medical assistance while maintaining patient privacy through local processing.

## Project Documents

### Core Documentation

- [Project Overview](project-info/ProjectOverview.md) - Main project description, goals, and tasks.
- [Technical Architecture](project-info/TechnicalArchitecture.md) - Detailed technical design and system components
- [Sample Code](project-info/SampleCode.md) - Code examples to help get started with implementation

### Task Documents
Tasks 1 to 4 will be prioritized because they relate to the primary goal of the project; they will have to be completed by the project deadline. Tasks 5 and 6 will be considered thorughout the project, but they will not be the primary focus. Nonetheless, given the learning potential, the team will seriously consider completing these tasks after the project deadline to facilitate integration into the Open Doctor project.

- **Project Management & API Design** - [Task 1 (Primary)](project-info/Task1_ProjectManager.md)
- **ICD Database Integration** - [Task 2 (Primary)](project-info/Task2_ICD.md)
- **Vector Database Implementation** - [Task 3 (Primary)](project-info/Task3_VectorDB.md)
- **LLM Integration** – [Task 4 (Primary)](project-info/Task4_LLMIntegration.md)
- **Testing & Quality Assurance** – [Task 5 (Secondary)](project-info/Task5_Testing.md)
- **Documntation** – [Task 6 (Secondary)](project-info/Task5_Testing.md)

## Project Deadline

The final project submission is due on May 25, 2025.

## Core Technologies

- **Python**: Primary programming language
- **FastAPI**: API framework
- **Vector Databases**: FAISS, Pinecone, Chroma DB, or similar
- **Sentence Transformers**: For creating embeddings
- **ICD-11 API**: WHO's International Classification of Diseases API ([Documentation](https://icd.who.int/docs/icd-api/APIDoc-Version2/), [API Reference](https://icd.who.int/icdapi/docs2/APIDoc-Version2/))
- **ICHI Browser**: WHO's International Classification of Health Interventions ([Browser](https://icd.who.int/dev11/l-ichi/en))
- **GitHub**: For code storage and collaboration
- **Docker**: For containerization and deployment

## Resources

- [ICD API Homepage](https://icd.who.int/icdapi)
- [ICD API Documentation v2.x](https://icd.who.int/docs/icd-api/APIDoc-Version2/)
- [ICD API Reference (Swagger)](https://icd.who.int/icdapi/docs2/APIDoc-Version2/)
- [Supported Classification Versions](https://icd.who.int/icdapi/docs2/SupportedClassifications/)
- [Clinical Table Search Service API for ICD-11](https://clinicaltables.nlm.nih.gov/apidoc/icd11_codes/v3/doc.html)
- [ICHI Browser](https://icd.who.int/dev11/l-ichi/en)

## Communication

Team members should establish regular communication channels and meeting schedules. All code and documentation should be maintained in a shared GitHub repository to facilitate collaboration and version control.

For questions about the [Open Doctor project](https://github.com/SEBK4C/OpenDoctor-Spec) or technical guidance, contact [SEBK4C](https://github.com/SEBK4C).

---

This project will enable the Open Doctor system to translate natural language descriptions of medical symptoms into standardized ICD codes and suggest appropriate medical interventions, forming a critical component of the broader goal to democratize access to medical knowledge while maintaining patient privacy. 