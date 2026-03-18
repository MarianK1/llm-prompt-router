import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import config


log = logging.getLogger(__name__)

class SemanticRouter:
    """ 
    Embedding-based Semantic Router.
    K-Nearest Neighbor to direct queries
    based on semantic proximity to 'simple' and 'complex' clusters
    """
    def __init__(self, simple_path = 'simple_embeddings.npy', complex_path = 'complex_embeddings.npy'):
        
        log.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # loading files with matrices
        self.simple_embeddings = np.load(simple_path)
        self.complex_embeddings = np.load(complex_path)
        
        
        
    def route(self, query: str) -> tuple[str, float, float]:
        # embed incoming query
        query_embedding = self._embed([query])
        
        # calculating similarity against every single example
        sim_simple = cosine_similarity(query_embedding, self.simple_embeddings)[0]
        sim_complex = cosine_similarity(query_embedding, self.complex_embeddings)[0]
        
        # maximum similarity score
        max_sim_s = float(np.max(sim_simple))
        max_sim_c = float(np.max(sim_complex))
        
        # routing
        if max_sim_s > max_sim_c:
            return "simple", max_sim_s, max_sim_c
        else:
            return "complex", max_sim_s, max_sim_c
    
    def _embed(self, texts: list[str]) -> np.ndarray:
        # convert to numpy returns an ndarray
        # normalize embeddings ensures L2 normalization
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        
if __name__ == "__main__":
    # testing and downloading for first run
    print("Running semantic router test...")
    simple_ex = ["say hello", "what time is it"]
    adv_ex = ["debug this memory leak", "explain quantum physics"]
    
    router = SemanticRouter()
    res = router.route("fix my python code")
    
    print(f"Test passed. Route: {res[0]}, Similarity for Simple: {res[1]:.3f}, Similarity for Complex: {res[2]:.3f}")