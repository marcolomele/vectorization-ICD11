# Preprocessing
- variable setting ✅
- check outliers with lengths -> fix top two entries ✅
- check for nan value distribution ✅

Quality assurance:
- gen descriptions longer ✅
- gen description is complete: if cover overview, causes, symptoms, transmission, diagnosis, treatment. ✅

Problem: many definitions incomplete, even more definitions missing. ✅

Solution 1: Text concatenation. ✅

Problem: low scores and TF-IDF wins. ✅

Solution 2: Llama3-OpenBio-70B. ✅

# LLM Data generation
Describe procedure.
Quality assurance:
- gen descriptions longer ✅
- gen description is complete: if cover overview, causes, symptoms, transmission, diagnosis, treatment. ✅
- clinical validity and relevance: random sample and comparison with ICD-11. ✅

# Comparative Data Analysis 
Goal: understand if LLM added information for embeddings.
- processing steps specific for analysis: lowercase, punctuation, lemmatisation. apply to both. ✅
- information density: length (characters, words, sentences), vocabulary richness, complexity, POS.
– identify information novely with n-gram overlap and BLUE/ROUGE scores 

If so, how does this information look like?
– name entity recognition
- topic modelling
- sentiment
- POS tagging

View across description sources, across categories.

# Hierarchy Analysis
ICD-11 is essentially a database structured as a tree. Hence, we can look at:
- Depth: descriptive statistics.
– Breadth: bushiness vs linearity of differnt paths (distribution of children per node at different levels).
- If nodes have multiple parents.

Explore both original and generated descriptions, across categories.

Note: this section is not directly related to NLP, hence keep minimal. 

# Preparing for Modelling
- concatenate intelligently new information from other columns.
- remove other columns and give essential data to modelling team.