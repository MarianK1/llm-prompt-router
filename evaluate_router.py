# implemented TFIDF-based router inside this file for comparison
import numpy as np
import time
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from semantic_router import SemanticRouter
from build_dataset import SIMPLE_EXAMPLES, COMPLEX_EXAMPLES

# 40 different queries dataset
LABELLED_TEST_SET: List[Tuple[str, str]] = [
    # 10 clearly simple
    ("What is the largest ocean on Earth?", "simple"),
    ("Can you tell me the current temperature in Berlin?", "simple"),
    ("How do you say 'goodbye' in Italian?", "simple"),
    ("Is a dolphin considered a mammal?", "simple"),
    ("Who painted the Mona Lisa?", "simple"),
    ("Subtract 45 from 100.", "simple"),
    ("Give me instructions on how to boil an egg.", "simple"),
    ("When was the Declaration of Independence signed?", "simple"),
    ("What is the currency used in Japan?", "simple"),
    ("Spell the word 'accommodation' correctly.", "simple"),

    # 10 clearly complex
    ("Troubleshoot a memory leak in a highly concurrent Golang backend service.", "complex"),
    ("Formulate a migration strategy for moving an on-premise Oracle cluster to Azure Cloud.", "complex"),
    ("Construct a Vue.js state management module using Pinia for a shopping cart.", "complex"),
    ("Discuss the sociological impacts of the Industrial Revolution on urban working classes.", "complex"),
    ("Architect a distributed message broker comparable to Apache Kafka.", "complex"),
    ("Evaluate the time complexity of the A* search algorithm using a binary min-heap.", "complex"),
    ("Draft a GitLab pipeline configuration for deploying a serverless AWS Lambda application.", "complex"),
    ("Explain the mathematical principles underlying the Transformer architecture's self-attention mechanism.", "complex"),
    ("Implement an OAuth 2.0 authorization code flow in a Spring Boot application.", "complex"),
    ("Compare the theoretical limitations of Bell's Theorem with local hidden-variable theories.", "complex"),

    # 10 complex short (edge cases)
    ("troubleshoot race condition", "complex"),
    ("explain ACID properties", "complex"),
    ("derive quadratic formula", "complex"),
    ("optimize Vue reactivity", "complex"),
    ("configure Azure Virtual Network", "complex"),
    ("implement merge sort", "complex"),
    ("analyze Red-Black tree", "complex"),
    ("bypass ASLR protection", "complex"),
    ("explain CAP theorem", "complex"),
    ("design micro-frontend", "complex"),
    
    # 10 ambiguous, nonsense or OOD (fallback to complex)
    ("xyzzy plugh twas brillig", "complex"),
    ("0987654321", "complex"),
    ("Translate this: explain ACID properties", "complex"),
    ("Instructions for cooking a distributed system", "complex"),
    ("Dónde está la biblioteca de los microservicios?", "complex"),
    ("What is the capital of race condition?", "complex"),
    ("qwertyuiop asdfghjkl", "complex"),
    ("How do I change my laptop's thermal paste?", "complex"),
    ("zxvcb nmasdf", "complex"),
    ("J'aime le segfault de C++", "complex"),
]

# inline tfidf router
class TfidfRouter:
    def __init__(self, simple_ex, complex_ex):
        self.vectorizer = TfidfVectorizer()
        
        # train vocab on all examples
        self.vectorizer.fit(simple_ex + complex_ex)
        
        # create centroids
        sim_vecs = self.vectorizer.transform(simple_ex)
        com_vecs = self.vectorizer.transform(complex_ex)
        
        # converse to numpy arrays
        self.simple_centroid = np.asarray(sim_vecs.mean(axis=0))
        self.complex_centroid = np.asarray(com_vecs.mean(axis=0))
        
    def route(self, query: str) -> str:
        q_vec = self.vectorizer.transform([query]).toarray()
        sim_s = cosine_similarity(q_vec, self.simple_centroid)[0][0]
        sim_c = cosine_similarity(q_vec, self.complex_centroid)[0][0]
        
        # fallback for OOD (all zeros in TF-IDF)
        if sim_s == 0.0 and sim_c == 0.0:
            return "complex"
        
        return "simple" if sim_s > sim_c else "complex"
    
def calculate_metrics(results: List[Dict]) -> Dict[str, float]:
    tp = sum(1 for r in results if r['expected'] == 'complex' and r['predicted'] == 'complex')
    tn = sum(1 for r in results if r['expected'] == 'simple' and r['predicted'] == 'simple')
    fp = sum(1 for r in results if r['expected'] == 'simple' and r['predicted'] == 'complex')
    fn = sum(1 for r in results if r['expected'] == 'complex' and r['predicted'] == 'simple')
    accuracy = (tp + tn) / len(results) if results else  0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def run_eval() -> None:
    print("Loading TF-IDF baselines...")
    tfidf_router = TfidfRouter(SIMPLE_EXAMPLES, COMPLEX_EXAMPLES)

    print("Loading Semantic Router v2...")
    semantic_router = SemanticRouter()

    tfidf_results = []
    semantic_results = []

    print("Evaluating 40 test queries...\n")
    for query, expected in LABELLED_TEST_SET:
        # v1: TF-IDF
        tf_label = tfidf_router.route(query)
        tfidf_results.append({"expected": expected, "predicted": tf_label})

        # v3: Dense Embeddings
        sem_label, _, _ = semantic_router.route(query)
        semantic_results.append({"expected": expected, "predicted": sem_label})

    # calculations
    tf_metrics = calculate_metrics(tfidf_results)
    sem_metrics = calculate_metrics(semantic_results)

    # printing report 
    print("=" * 65)
    print(" ACCURACY BENCHMARK: TF-IDF vs. DENSE-EMBEDDINGS")
    print("=" * 65)

    print(f"{'Metric':<15} | {'TF-IDF (V1)':<15} | {'Semantic (V2)':<15} | {'Delta':<10}")
    print("-" * 65)

    print(f"{'Accuracy':<15} | {tf_metrics['accuracy']:.2%} {'':<8} | {sem_metrics['accuracy']:.2%} {'':<8} | {sem_metrics['accuracy'] - tf_metrics['accuracy']:+.2%}")
    print(f"{'Precision (C)':<15} | {tf_metrics['precision']:.2%} {'':<8} | {sem_metrics['precision']:.2%} {'':<8} | {sem_metrics['precision'] - tf_metrics['precision']:+.2%}")
    print(f"{'Recall (C)':<15} | {tf_metrics['recall']:.2%} {'':<8} | {sem_metrics['recall']:.2%} {'':<8} | {sem_metrics['recall'] - tf_metrics['recall']:+.2%}")
    print(f"{'F1 Score (C)':<15} | {tf_metrics['f1']:.3f} {'':<9} | {sem_metrics['f1']:.3f} {'':<9} | {sem_metrics['f1'] - tf_metrics['f1']:+.3f}")
    print("=" * 65)

if __name__ == "__main__":
    run_eval()