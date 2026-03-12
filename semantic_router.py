import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import config

class SemanticRouter:
    """ 
    Embedding-based Semantic Router.
    TF-IDF Vectorization and Cosine Similarity to direct queries
    based on semantic proximity to 'simple' and 'complex' clusters
    """
    def __init__(self):
        self.simple_examples = [
            "Translate 'hello' to French.",
            "What is the capital city of Albania?",
            "Summarize this email in one sentence:...",
            "What time is it in Bogota now?",
            "Give me a quick recipe for pancakes."
        ]
        
        self.complex_examples = [
            "Debug this Python script and explain in detail why it throws this memory error.",
            "Analyze the logic behind quantum computing and explain it to a high-schooler in a few words.",
            "Write a detailed architectural plan for migrating a Postgres database to AWS with distinct steps of its architecture.",
            "Why did the Roman Empire fall? Compare the economic and military factors.",
            "Write a React component that manages state for a multi-step checkout form."
        ]
        
        # initializing and training vectorizer on known data
        self.vectorizer = TfidfVectorizer(stop_words = 'english')
        all_examples = self.simple_examples + self.complex_examples
        
        self.vectorizer.fit(all_examples)
        
        # creating mathematical centroids or avg vectors for both sides
        simple_matrix = self.vectorizer.transform(self.simple_examples)
        self.simple_centroid = np.mean(simple_matrix.toarray(), axis = 0).reshape(1, -1)
        
        complex_matrix = self.vectorizer.transform(self.complex_examples)
        self.complex_centroid = np.mean(complex_matrix.toarray(), axis = 0).reshape(1, -1)
        
        
    def small_model(self, query: str) -> dict:
        return {"model": config.models["small"]["name"], "cost_tier": "low", "query": query}
    
    def adv_model(self, query: str) -> dict:
        return {"model": config.models["advanced"]["name"], "cist_tier": "high", "query": query}
    
    def route_query(self, query: str) -> dict:
        # convert incoming query into vector
        query_vector = self.vectorizer.transform([query]).toarray()
        
        # calculate cosine similarity to centroids
        sim_to_simple = cosine_similarity(query_vector, self.simple_centroid)[0][0]
        sim_to_complex = cosine_similarity(query_vector, self.complex_centroid)[0][0]
        
        # routing logic
        if sim_to_complex > sim_to_simple or len(query.split()) > 100:
            return self.adv_model(query)
        elif sim_to_simple > sim_to_complex:
            return self.small_model(query)
        else:
            # some sort of possible tiebreaker in OOD cases
            if len(query.split()) > config.LEN_THRESHOLD:
                return self.adv_model(query)
            return self.small_model(query)
        
if __name__ == "__main__":
    router = SemanticRouter()
    
    test_queries = [
        "What is 5 plus 5?", # unseen simple
        "Can you build a scalable microservice architecture through Kubernetes and gRPC?", # complex
        "Explain performance tradeoffs in an LLM cascade" # complex
    ]
    
    print(f"{'Query':<80} | {'Assigned Model'}")
    print("-"*100)
    for q in test_queries:
        result = router.route_query(q)
        print(f"{q:<80} | {result['model']}")