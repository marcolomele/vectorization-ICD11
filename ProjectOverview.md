# ICD Database Vectorization for Open Doctor Project

## Project Summary

This project aims to develop a vector database system for the International Classification of Diseases (ICD) database maintained by the World Health Organization. The core functionality will allow for natural language queries of medical symptoms and conditions to retrieve the appropriate ICD codes. The vision is to facilitate self-diagnosis, giving people more control of their cures and helping doctors to treat more people. 

### Data

The [ICD database](https://icd.who.int/en) is the internationally recognized standard for classifying diseases, symptoms, and medical conditions. Every country has their own version of this database, but the global standard is maintained in English by the WHO. The database contains codes for all known diseases and medical conditions, providing a standardized way to classify and communicate about medical diagnoses.

Currently, the ICD database is accessible through a [REST API](https://icd.who.int/icdapi) but lacks efficient natural language query capabilities and vector search functionality. This project aims to fill this gap by creating a vector representation of the ICD database to enable more intuitive searching based on patient-described symptoms.

### Project Goals

1. Create a vector database of the [ICD-11](https://icd.who.int/docs/icd-api/APIDoc-Version2/) disease codes using symptom descirptions. 
2. Develop an API that allows natural language queries to find relevant ICD codes.
3. Integrate an LLM to enable conversational interactions and improve accessibility.

#### Stretch Goals (if time permits)
1. Build a vector database for medical interventions using [ICHI](https://icd.who.int/dev11/l-ichi/en).
2. Develop matching algorithms to link diagnosis codes with appropriate intervention codes.

## Technical Architecture
### Fundamental
The project will utilize the following technologies:
- **Vector Database**: Using an embedding model to convert medical terms and descriptions into vector representations
- **FastAPI**: For creating a REST API that serves vector search results
- **ICD-11 API**: The WHO's official API for accessing ICD codes and descriptions ([Documentation](https://icd.who.int/docs/icd-api/APIDoc-Version2/), [API Reference](https://icd.who.int/icdapi/docs2/APIDoc-Version2/))
- **Clinical Table Search Service**: Alternative API for ICD-11 ([Documentation](https://clinicaltables.nlm.nih.gov/apidoc/icd11_codes/v3/doc.html))

See [technical architecture](TechnicalArchitecture.md) for full overview.

### Nice-to-haves
The project will attempt to include the following technical components. However, these are of secondary importance, and might be deliverared at a later date if time and resources allow it.
- **Testing Framework**: Pytest for comprehensive testing of all components
- **GitHub Actions**: For continuous integration and test automation
- **Docker**: For containerization and easy deployment

## Project Tasks

The following tasks need to be completed for the project:

1. **Project Management & API Design** - [Task 1](Task1_ProjectManager.md)
   - Coordinate team activities and track deliverables
   - Design API endpoints and response formats
   - Integrate individual components
   - Manage project timeline and milestones

2. **ICD Database Integration** - [Task 2](Task2_ICD.md)
   - Study and document ICD-11 database structure
   - Implement data extraction from ICD-11 API
   - Document ICD code hierarchy and relationships
   - Create data validation and cleaning pipelines

3. **Vector Database Implementation** - [Task 3](Task3_VectorDB.md)
   - Research and select vector database technology
   - Implement vector database solution
   - Create embedding pipeline for medical terms
   - Optimize vector search algorithms
   - Implement caching and performance optimizations

4. **LLM Integration** – [Task 4](Task4_LLMIntegration.md)
   - Research and select appropriate LLM solution
   - Implement LLM integration with vector database
   - Create conversational interface
   - Optimize prompt engineering for medical context
   - Implement response formatting and validation

5. **Testing & Quality Assurance** – [Task 5 (Optional)](Task5_Testing.md)
   - Create comprehensive test cases
   - Implement automated testing pipeline
   - Ensure deterministic test results
   - Document test coverage and results
   - Perform performance testing and optimization

6. **Documentation & Integration** – [Task 6 (Optional)](Task6_Documentation.md)
   - Create technical documentation
   - Prepare project report
   - Ensure system can be integrated with Open Doctor
   - Develop user guides and examples
   - Create API documentation

## Timeline

- **Week 1** (April 19-25): Research and planning
  - Study ICD database structure
  - Study and select vector database technology
  - Design system architecture

- **Week 2** (April 26-May 2): Core implementation
  - Extract ICD codes and descriptions
  - Implement vector database
  - Create basic API endpoints

- **Week 3** (May 3-9): Integration and testing
  - Connect all components
  - Optimize search performance

- **Week 4** (May 10-16): Finalization
  - Finalize code
  - Prepare presentation
  - Create final report

- **Final Submission**: May 25, 2025

## Success Criteria

The project will be considered successful if:

1. The system can accurately match natural language descriptions of symptoms to relevant ICD codes
2. The LLM integration provides clear, conversational responses to medical queries, along with ICD codes. 

## Communication and Collaboration
- **GitHub**: All code and documentation will be stored in a GitHub repository.
- **WhatsApp** and **Google Meet**: Team communication.

## Resources

- [ICD API Homepage](https://icd.who.int/icdapi)
- [ICD API Documentation v2.x](https://icd.who.int/docs/icd-api/APIDoc-Version2/)
- [ICD API Reference (Swagger)](https://icd.who.int/icdapi/docs2/APIDoc-Version2/)
- [Supported Classification Versions](https://icd.who.int/icdapi/docs2/SupportedClassifications/)
- [Clinical Table Search Service API for ICD-11](https://clinicaltables.nlm.nih.gov/apidoc/icd11_codes/v3/doc.html)
- [ICHI Browser](https://icd.who.int/dev11/l-ichi/en)

## Contact

If you have questions about the [Open Doctor project](https://github.com/SEBK4C/OpenDoctor-Spec) or need technical guidance, please contact Sebastian Graf.