import os
import json
from tqdm import tqdm
import pandas as pd

def extract_icd_text_data(json_folder, i):
    records = []

    for filename in tqdm(os.listdir(json_folder), desc=f"Processing files in chapter {i}..."):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(json_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Initialize record with basic fields that might be present
        record = {}
        
        # Core identifiers (always try to get these)
        record["id"] = data.get("@id", "").split("/")[-1]
        record["code"] = data.get("code", "")
        
        # Optional basic metadata
        if "title" in data:
            record["title"] = data.get("title", {}).get("@value", "")
        if "browserUrl" in data:
            record["browser_url"] = data.get("browserUrl", "")
        if "classKind" in data:
            record["class_kind"] = data.get("classKind", "")
        
        # Optional detailed information
        if "definition" in data:
            record["definition"] = data.get("definition", {}).get("@value", "")
        if "fullySpecifiedName" in data:
            record["fully_specified_name"] = data.get("fullySpecifiedName", {}).get("@value", "")

        # Hierarchical relationships (if present)
        if "parent" in data:
            record["parent"] = "; ".join([p.split("/")[-1] for p in data.get("parent", [])])
        if "child" in data:
            record["children"] = "; ".join([c.split("/")[-1] for c in data.get("child", [])])

        # Inclusions (if present)
        if "inclusion" in data:
            inclusions = []
            for inclusion in data["inclusion"]:
                label = inclusion.get("label", {}).get("@value", "")
                if label:
                    inclusions.append(label)
            record["inclusions"] = "; ".join(inclusions)

        # Exclusions (if present)
        if "exclusion" in data:
            exclusions = []
            exclusion_refs = []
            for excl in data["exclusion"]:
                label = excl.get("label", {}).get("@value", "")
                if label:
                    exclusions.append(label)
                    if "foundationReference" in excl:
                        exclusion_refs.append(f"{label}: {excl['foundationReference']}")
            record["exclusions"] = "; ".join(exclusions)
            if exclusion_refs:
                record["exclusion_references"] = "; ".join(exclusion_refs)

        # Foundation children (if present)
        if "foundationChildElsewhere" in data:
            foundation_children = []
            foundation_child_refs = []
            for child in data["foundationChildElsewhere"]:
                label = child.get("label", {}).get("@value", "")
                if label:
                    foundation_children.append(label)
                    if "foundationReference" in child:
                        foundation_child_refs.append(f"{label}: {child['foundationReference']}")
            record["foundation_children"] = "; ".join(foundation_children)
            if foundation_child_refs:
                record["foundation_child_references"] = "; ".join(foundation_child_refs)

        # Index terms (if present)
        if "indexTerm" in data:
            index_terms = []
            index_term_refs = []
            for term in data["indexTerm"]:
                label = term.get("label", {}).get("@value", "")
                if label:
                    index_terms.append(label)
                    if "foundationReference" in term:
                        index_term_refs.append(f"{label}: {term['foundationReference']}")
            record["index_terms"] = "; ".join(index_terms)
            if index_term_refs:
                record["index_term_references"] = "; ".join(index_term_refs)

        # Postcoordination scales (if present)
        if "postcoordinationScale" in data:
            postcoord_scales = []
            for scale in data["postcoordinationScale"]:
                scale_info = {
                    "axis_name": scale.get("axisName", "").split("/")[-1],
                    "required": scale.get("requiredPostcoordination", ""),
                    "allow_multiple": scale.get("allowMultipleValues", ""),
                    "entities": "; ".join([e.split("/")[-1] for e in scale.get("scaleEntity", [])])
                }
                postcoord_scales.append(str(scale_info))
            record["postcoordination_scales"] = " || ".join(postcoord_scales)

        # Related entities (if present)
        if "relatedEntitiesInPerinatalChapter" in data:
            record["related_entities"] = "; ".join([e.split("/")[-1] for e in data["relatedEntitiesInPerinatalChapter"]])

        # Construct full text only from available fields
        full_text_parts = []
        for field in ["title", "definition", "fully_specified_name", "inclusions", 
                     "exclusions", "foundation_children", "index_terms"]:
            if field in record and record[field]:
                full_text_parts.append(record[field])
        record["full_text"] = " ".join(full_text_parts)

        records.append(record)

    # Create DataFrame with all possible columns, filling missing values with empty strings
    df = pd.DataFrame(records)
    df = df.fillna("")
    return df

def get_json_folders(base_folder):
    folders = [os.path.join(base_folder, f) for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    folders = sorted(folders)
    return folders

def data_to_df(json_folders):
    dfs = []
    for i, folder in enumerate(json_folders):
        df = extract_icd_text_data(folder, i)
        dfs.append(df)
    return pd.concat(dfs)

def main():
    base_folder = input("Enter the base folder name: ")
    json_folders = get_json_folders(base_folder)
    df = data_to_df(json_folders)
    df.to_csv("icd_text_data_raw.csv", index=False)

if __name__ == "__main__":
    main()