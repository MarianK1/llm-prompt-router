import json
import datetime
import numpy as np
from sentence_transformers import SentenceTransformer

# datasets

# 60 simple examples
SIMPLE_EXAMPLES = [
    "Who invented the telephone?", "What is the capital of Japan?",
    "What is photosynthesis?", "Define entropy",
    "How many feet in a mile?", "Convert 100 USD to EUR",
    "When did WW2 end?", "How many days in a leap year?",
    "Tell me a joke", "How are you?",
    "What's a good movie to watch?", "How do I hard boil an egg?",
    "How do I restart my computer?", "Write a haiku about rain",
    "Give me a fun fact", "How do you say hello in French?",
    "Translate 'thank you' to Japanese", "Is the sun a star?",
    "Do penguins live in the Arctic?", "Summarise the plot of Romeo and Juliet in one sentence",
    "What is the deepest ocean trench?", "Who painted the Mona Lisa?",
    "Explain the water cycle simply.", "What does DNA stand for?",
    "How many ounces are in a cup?", "Convert 32 Fahrenheit to Celsius.",
    "What year did the Titanic sink?", "How many weeks are in a year?",
    "Tell me a knock-knock joke.", "What's your favorite color?",
    "Recommend a book for a teenager.", "How do I tie a tie?",
    "How do I clear my browser cache?", "Write a short poem about a cat.",
    "Tell me something interesting.", "How do you say 'good morning' in Spanish?",
    "Translate 'I love you' into German.", "Is a tomato a fruit or a vegetable?",
    "Do sharks have bones?", "Give me a one-sentence summary of the Harry Potter series.",
    "What is the largest planet in our solar system?", "Who wrote 'To Kill a Mockingbird'?",
    "What is the boiling point of water in Fahrenheit?", "Define 'onomatopoeia'.",
    "How many millimeters are in a meter?", "Convert 5 kilometers to miles.",
    "When was the Declaration of Independence signed?", "How many days are in February during a non-leap year?",
    "Tell me a funny story.", "What time is it right now?",
    "Suggest a good podcast to listen to.", "How do I make a simple grilled cheese sandwich?",
    "How do I connect to Wi-Fi?", "Write a two-line rhyme about spring.",
    "Share a random piece of trivia.", "Is 7 prime?",
    "What colour is the sky?", "What is 2 plus 2?",
    "Is water wet?", "What day is it today?"
]
# 60 complex examples
COMPLEX_EXAMPLES = [
    "Design a rate limiter for a distributed API", "Architect a real-time recommendation system",
    "My Kubernetes pod keeps OOMKilling, diagnose it", "Why does this React component cause infinite re-renders?",
    "Derive the backpropagation equations for a 3-layer MLP", "Prove that sqrt(2) is irrational",
    "What is a timing side-channel attack?", "Explain the vanishing gradient problem and three solutions",
    "What is the difference between RLHF and DPO?", "Refactor this O(n²) algorithm to O(n log n)",
    "Identify race conditions in this async code", "What are the GDPR implications of storing embeddings of user queries?",
    "Analyse the trolley problem from a utilitarian vs deontological perspective", "What causes protein misfolding in prion diseases?",
    "Analyse the competitive moat of a vertically integrated AI company", "What are the second-order effects of the Fed raising interest rates?",
    "Write a 30-day learning plan to go from zero to deploying a transformer model", "How would I migrate a monolith to microservices with zero downtime?",
    "Design a highly available distributed cache system like Redis.", "Architect a scalable pub/sub messaging system.",
    "A production database query is suddenly slow. Outline your troubleshooting steps.", "Explain the intricacies of event loop blocking in Node.js.",
    "Derive the formula for the volume of a sphere using calculus.", "Prove the Pythagorean theorem using geometric arguments.",
    "Detail the mechanics of a Cross-Site Request Forgery (CSRF) exploit and mitigation strategies.", "Explain the role of the attention mechanism in Transformer models.",
    "Compare and contrast the performance characteristics of B-trees and LSM-trees.", "Optimize a given piece of Python code that is bottlenecked by a global interpreter lock (GIL).",
    "Find the deadlocks in this multithreaded Java application.", "Discuss the ethical considerations of using biased datasets to train predictive policing algorithms.",
    "Compare the political philosophies of Hobbes, Locke, and Rousseau on the social contract.", "Describe the detailed molecular mechanism of CRISPR-Cas9 gene editing.",
    "Evaluate the potential impact of generative AI on the future of the software engineering job market.", "Analyze the geopolitical implications of a global shift towards renewable energy sources.",
    "Create a comprehensive disaster recovery plan for a cloud-native application.", "Outline the steps to implement zero-trust architecture in an enterprise network.",
    "Design an architecture for a global scale video streaming service.", "How would you design an event-driven microservices architecture?",
    "Troubleshoot a network latency issue between two AWS regions.", "Explain how garbage collection works in Java and how to tune it for low-latency applications.",
    "Derive the fundamental theorem of calculus.", "Provide a formal proof by induction for the sum of the first n integers.",
    "Explain how a SQL injection attack works and how to use parameterized queries to prevent it.", "Discuss the architecture and training process of a Generative Adversarial Network (GAN).",
    "Explain the CAP theorem and its implications for distributed database design.", "Refactor a legacy monolithic application component into a separate microservice.",
    "Identify memory management issues in this C++ program using Valgrind output.", "Analyze the legal implications of using copyrighted material in training datasets for AI models.",
    "Compare Keynesian and classical economic theories regarding government intervention in the economy.", "Explain the process of cellular respiration in detail.",
    "Develop a strategic plan for a startup entering a highly saturated market.", "Analyze the long-term economic consequences of a universal basic income policy.",
    "Create a roadmap for migrating an on-premise data warehouse to a cloud-based solution.", "Outline a strategy for scaling an engineering team from 10 to 100 developers.",
    "Why is sorting hard?", "What is consciousness?",
    "Is time travel mathematically possible?", "What defines art?",
    "Why does anything exist rather than nothing?", "How does the economy work?"
]

# building engine
def build_and_save():
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Encoding {len(SIMPLE_EXAMPLES)} simple examples...")
    simple_embs = model.encode(SIMPLE_EXAMPLES, convert_to_numpy = True, normalize_embeddings = True)
    
    print(f"Encoding {len(COMPLEX_EXAMPLES)} simple examples...")
    complex_embs = model.encode(COMPLEX_EXAMPLES, convert_to_numpy = True, normalize_embeddings = True)
    
    print("Saving matrices in .npy files...")
    np.save('simple_embeddings.npy', simple_embs)
    np.save('complex_embeddings', complex_embs)
    
    print("Generating .json...")
    manifest = {
        "built_at": datetime.date.today().isoformat(),
        "model": "all-MiniLM-L6-v2",
        "simple_examples_count": len(SIMPLE_EXAMPLES),
        "complex_examples_count": len(COMPLEX_EXAMPLES),
        "embedding_dim": simple_embs.shape[1],
        "files": [
            "simple_embeddings.npy",
            "complex_embeddings.npy"
        ],
        "notes": "Pre-computed online. Run the '.py' file to regenerate. See README for methodology."
    }
    
    with open('dataset_manifest.json', 'w') as f:
        json.dump(manifest, f, indent = 2)
    
    print("Build complete. Artifacts generated successfully.")
    
if __name__ == "__main__":
    build_and_save()