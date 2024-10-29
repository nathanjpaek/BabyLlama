import nltk
import spacy
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.parse import CoreNLPParser
from nltk.corpus import stopwords
from collections import defaultdict

# Ensure nltk data is downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Load Spacy model with coreference resolution and POS tagging capabilities
nlp = spacy.load("en_core_web_sm")

def calculate_type_token_ratio(text):
    words = word_tokenize(text)
    unique_words = set(words)
    ttr = len(unique_words) / len(words)
    return ttr

def calculate_avg_sentence_depth(text):
    sentences = sent_tokenize(text)
    parser = CoreNLPParser()
    depths = []
    for sentence in sentences:
        try:
            parse_tree = next(parser.parse(sentence.split()))
            depths.append(parse_tree.height())
        except Exception as e:
            continue  # Skip sentences that can't be parsed
    avg_depth = sum(depths) / len(depths) if depths else 0
    return avg_depth

def calculate_coreference_resolution(text):
    doc = nlp(text)
    coref_counts = defaultdict(int)
    for cluster in doc._.coref_clusters:
        coref_counts[cluster.main.text] += len(cluster.mentions)
    return coref_counts

def calculate_pos_distribution(text):
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    pos_counts = Counter(tag for word, tag in pos_tags)
    total = sum(pos_counts.values())
    pos_distribution = {tag: count / total for tag, count in pos_counts.items()}
    return pos_distribution

def analyze_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    ttr = calculate_type_token_ratio(text)
    avg_sentence_depth = calculate_avg_sentence_depth(text)
    coref_res = calculate_coreference_resolution(text)
    pos_dist = calculate_pos_distribution(text)

    return {
        "Type-Token Ratio": ttr,
        "Average Sentence Depth": avg_sentence_depth,
        "Coreference Resolution Counts": coref_res,
        "POS Distribution": pos_dist
    }

# Example Usage
file_paths = [
    './data/babylm_10M_clean',
    './data/childes_10M',
    './data/gutenberg_10M',
    './data/tinystories'
]

results = {}
for file_path in file_paths:
    results[file_path] = analyze_text(file_path)

# Output results for each file
for file_path, metrics in results.items():
    print(f"Metrics for {file_path}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print("\n")
