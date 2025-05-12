# Preprocessing
- variable setting
- check outliers with lengths -> fix top two entries
- check for nan value distribution

Quality assurance:
- gen descriptions longer
- gen description is complete: if cover overview, causes, symptoms, transmission, diagnosis, treatment.

Problem: many definitions incomplete, even more definitions missing.

Solution: Llama3-OpenBio-70B.

# LLM Data generation
Describe procedure.
Quality assurance:
- gen descriptions longer
- gen description is complete: if cover overview, causes, symptoms, transmission, diagnosis, treatment.
- clinical validity and relevance: random sample and comparison with ICD-11.

# Comparative Data Analysis 
Goal: understand if LLM added information for embeddings.
- processing steps specific for analysis: lowercase, punctuation, lemmatisation. apply to both.
- information density: length (characters, words, sentences), vocabulary richness, complexity.
– identify information novely with n-gram overlap and BLUE/ROUGE scores 

If so, how does this information look like?
– name entity recognition
- topic modelling
- sentiment
- POS tagging

View across description sources, across categories.

# Hierarchy Analysis
- depth: pre each node
- breadth: bushiness vs linearity of differnt paths (distribution of children per node at different levels).
- multiple parents 

view across description sources, across categories


