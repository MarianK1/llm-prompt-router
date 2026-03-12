import logging
from typing import Dict
import config

# config log to simulate observability
logging.basicConfig(level = logging.INFO, format = '%(levelname)s - %(message)s')

class QueryRouter:
    """
    Taxonomy-guided selector that is lightweight.
    Queries are evaluated based on lexical intent and length signals to optimize
    cost-latency border before routing them to the appropriate LLM.
    
    """
    def __init__(self):
        self.threshold = config.ROUTING_THRESHOLD
        self.len_limit = config.LEN_THRESHOLD
        self.signals = config.SIGNALS
        self.models = config.models
        
    def _route_to_small(self, query: str, score: int) -> Dict:
        logging.info(f"[Routed to Small] Score: {score} | Query: '{query[:30]}'")
        return {
            "model": self.models["small"]["name"],
            "cost_tier": "low",
            "latency_tier": "low",
            "query": query
        }
        
    def _route_to_advanced(self, query: str, score: int) -> Dict:
        logging.info(f"[Routed to Advanced] Score: {score} | Query: '{query[:30]}'")
        return {
            "model": self.models["advanced"]["name"],
            "cost_tier": "high",
            "latency_tier": "high",
            "query": query
        }
        
    def get_route(self, query: str) -> Dict:
        """ Calculates complexity score and directs query to the right model """
        score = 0
        
        # signal 1: syntax (length as proxy for attn/latency)
        if len(query.split()) > self.len_limit:
            score += 1
            
        # signal 2: lexicon (taxonomy mapping)
        if self.signals["complex"].search(query):
            score += 2
        if self.signals["simple"].search(query):
            score -= 1
            
        # decision boundary
        if score >= self.threshold:
            return self._route_to_advanced(query, score)
        else:
            return self._route_to_small(query, score)

        