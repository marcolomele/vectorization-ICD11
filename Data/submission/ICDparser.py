import os
import json
import time
import requests
import collections
from tqdm import tqdm

# ========== CONFIGURATION ==========
CLIENT_ID = "6afc7b11-86c2-40f5-82d7-39c6c5869dec_38c25903-b150-4472-80cf-58648427fd58"
CLIENT_SECRET = "4kfkQUPOf4fBozRuZIMYsfff4n22mxUg/7dz85YXMqM="
TOKEN_ENDPOINT = 'https://icdaccessmanagement.who.int/connect/token'
ROOT_ENTITY_URI = 'https://id.who.int/icd/release/11/2025-01/mms'
MMS_BASE_URI = 'https://id.who.int/icd/release/11/2025-01/mms/'

SLEEP_TIME = 0.01  # To avoid rate limiting
# ===================================

def get_access_token():
    payload = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'scope': 'icdapi_access',
        'grant_type': 'client_credentials'
    }
    response = requests.post(TOKEN_ENDPOINT, data=payload)
    response.raise_for_status()
    token_data = response.json()
 
    return token_data['access_token']

def crawl_icd11():
    print("\nGetting access token...")
    token = get_access_token()

    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json',
        'Accept-Language': 'en',
        'API-Version': 'v2'
    }

    # Fetch root structure
    response = requests.get(ROOT_ENTITY_URI, headers=headers)
    root_data = response.json()
    root_children = root_data.get('child', [])

    # Get root IDs and save them
    root_ids = [uri.split('/')[-1] for uri in root_children]
    with open("crawling_results_25/MMS_ROOTS.json", 'w') as f:
        json.dump(root_ids, f)

    # Fetch and display chapter titles
    for entity_id in root_ids:
        uri = f'{MMS_BASE_URI}{entity_id}'
        response = requests.get(uri, headers=headers)
        if response.status_code == 200:
            data = response.json()
            title = data.get('title', {}).get('@value', 'Unknown Title')
            code = data.get('code', "")
            print(f"- {code} {title} (ID: {entity_id})")
    
    for i in range(1):
        # if i == 0:
        #     EXTRACT_CHAPTER_NUMBER = 27
        #     OUTPUT_DIR = f'crawling_results_25/icd11_crawled_entities_iterative_CH_V'
        # else:
        #     EXTRACT_CHAPTER_NUMBER = 28
        #     OUTPUT_DIR = f'crawling_results_25/icd11_crawled_entities_iterative_CH_X'

        EXTRACT_CHAPTER_NUMBER = 6

        print(f"\nCrawling CHAPTER {EXTRACT_CHAPTER_NUMBER}...")
        OUTPUT_DIR = f'crawling_results_25/icd11_crawled_entities_iterative_CH_{str(EXTRACT_CHAPTER_NUMBER)}'
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # user_input = input("\nConfirm crawling process? (yes/no): ").lower().strip()
        # if user_input != 'yes':
        #     print("\nCrawling cancelled by user.")
        #     return

        # Initialize queue with root IDs
        chapter_to_process = root_ids[EXTRACT_CHAPTER_NUMBER-1]
        queue = collections.deque([chapter_to_process])
        visited = set([chapter_to_process])
        total_processed = 0
        
        # Iterative BFS traversal
        with tqdm(desc="Fetching entities", unit=" entities") as pbar:
            while queue:
                entity_id = queue.popleft()
                
                # Fetch entity data
                uri = f'{MMS_BASE_URI}{entity_id}'
                try:
                    response = requests.get(uri, headers=headers)
                    if response.status_code != 200:
                        print(f"\n!!! Failed to fetch {uri}")
                        continue
                    
                    data = response.json()
                    child_uris = data.get('child', [])
                    for child_uri in child_uris:
                        child_id = child_uri.split("/")[-1]
                        if child_id not in visited:
                            queue.append(child_id)
                            visited.add(child_id)
                    
                    # Save entity data
                    with open(os.path.join(OUTPUT_DIR, f"{entity_id}.json"), 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    
                    # Update progress
                    total_processed += 1
                    pbar.update(1)
                    
                    # Sleep to avoid rate limiting
                    time.sleep(SLEEP_TIME)
                    
                except Exception as e:
                    print(f"!!! Error processing {uri}: {str(e)}")
        
        print(f"\n FINISHED. {total_processed} MMS linearized entities saved to folder '{OUTPUT_DIR}\n\n")

def main():
    crawl_icd11()

if __name__ == "__main__":
    main()