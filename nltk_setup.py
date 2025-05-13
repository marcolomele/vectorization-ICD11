import nltk
import os

def setup_nltk():
    """Setup NLTK data if not already downloaded."""
    # Define required NLTK data
    required_data = {
        'punkt': 'tokenizers/punkt',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'stopwords': 'corpora/stopwords'
    }
    
    # Check if NLTK data directory exists
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    
    # Check and download each required component
    for component, path in required_data.items():
        try:
            nltk.data.find(path)
            print(f"NLTK {component} is already downloaded")
        except LookupError:
            print(f"Downloading NLTK {component}...")
            nltk.download(component)
            print(f"NLTK {component} downloaded successfully")

if __name__ == "__main__":
    setup_nltk() 