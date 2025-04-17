# ICD-11 CSV Columns Description
Check [icd11_data_raw.csv](icd11_data_raw.csv) for the original columns.

## Core Fields
- `id`: Unique identifier for the ICD-11 entity
- `code`: Official ICD-11 code (e.g., "A00-B99"); if its missing, it means it is an intermediate node. 
- `title`: Official title of the disease/condition
- `browser_url`: Link to the WHO ICD-11 browser
- `class_kind`: Type of classification (e.g., "chapter", "block", "category")

## Descriptive Fields
- `definition`: Clinical definition of the condition
- `fully_specified_name`: Complete medical term
- `inclusions`: List of conditions included in this category
- `exclusions`: List of conditions excluded from this category
- `index_terms`: Alternative terms or synonyms

## Hierarchical Relationships
- `parent`: Parent entity ID(s)
- `children`: Child entity ID(s)
- `foundation_children`: Related entities from foundation layer
- `related_entities`: Related entities in perinatal chapter

## References
- `exclusion_references`: References to excluded conditions
- `foundation_child_references`: References to foundation children
- `index_term_references`: References to index terms

## Postcoordination
- `postcoordination_scales`: Information about postcoordination axes

## Combined Text
- `full_text`: Combined text field for semantic search 