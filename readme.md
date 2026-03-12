# Cost/Latency Prompt Router for LLMs
This repository shows two distinct implementations of LLM routing, demonstrating progression
from a rapid heuristic baseline to an embedding-based semantic classifier, which is more robust.

## Architecture Overview
### Heuristic Router
* **Approach:** Deterministic, taxonomy-guided selector that uses regex keywords and query length
* **Signals Used:** 
    * **Lexical:** Presence of action-oriented words (*analyze, debug*) signals the need for high reasoning, ergo a higher model. Informational verbs (*translate, summarize*) signal low reasoning, ergo the smaller model.
   * **Syntactic:** Queries over 20 words might require larger context windows and deeper attention mechanisms, which signal the route towards the higher model.
* **Tradeoffs:** Zero latency and zero compute cost. *However*, it is very brittle. 5-word math puzzles can be highly complex, but this naive heuristic might misroute it towards the less-capable model (which might not be able to solve it).

### Semantic Router
* **Approach:** Embedding-based Semantic Router through scikit-learn library
* **Signals Used:** 
   * **Latent-Space Proximity:** To fix for the naivety of the first heuristic router, this semantic one uses *'TfidfVectorizer'* to map queries into one sparse vector space. Based on our baseline data, it calculates mathematical centroids for the *"Simple"* and *"Complex"* query clusters. When a new query appears, it calculates Cosine Similarity to both centroids, routing the query to the geometric cluster it is closest to.
* **Tradeoffs:** A bit higher computing cost needed compared to regex, but is significantly more capable against OOD queries while remaining clearly cheaper and faster than an "LLM-as-a-judge" cascading approach.

---

## **Theoretical Background and Inspiration**
The architecture of this system got inspired by recent industry research in LLM routing:
* **Taxonomy-Guided Selection:** Similar to SELECT-THEN-ROUTE (StR) framework, categorizing inputs into semantic classes before routing reduces significantly the decision space, which has been shown to reduce inference costs by up to 4 times without sacrificing accuracy.
* **Constrained Optimization:** This architecture treats routing as a problem with multiple objectives, seeking to maximize *Cost-Accuracy Frontier* by treating length and semantic proximity as proxy signals for computing cost and latency.

---

## Testing Safely
Deploying a router blindly can lead to massive cost spikes or decay in response quality. I would test for this through a 3-phase rollout:

1. **Offline Evaluation:** Run the router against a dataset of 1,000 hand-labelled prompts to measure *Precision* and *Recall*. The goal is to tune Cosine Similarity thresholds to penalize ***false negatives***, as this erodes user trust. *(see evaluate_router.py for a simulated phase 1 eval)*
2. **Shadow Routing:** Through a dark launch, the semantic router could be deployed to production, but it will **not** route traffic. All queries will still follow the previous model of routing. Our new version only logs what it **would** have done. We review the logs to ensure the new router does not trigger any cost-spikes or misclassifications of queries in real-world conditions.
3. **Canary Release:** A form of A/B testing, where 10% of live traffic passes through the new router, while the other 90% remains at the baseline. **Average Latency(ms)**, **Cost per 1k Queries ($)**, and **User Acceptance Rate** (thumbs up/down, regenerate requests, etc.). We scale to 50-50, then deploy the new router fully if quality holds and costs do not spike.

---

## Failure Modes & Mitigations

**First Failure Mode:** *"Prompt Padding" Cost Spike*
Power users discover the router's semantic signals. Complex phrasing gets used to simple tasks (e.g., 'analyze this email thread in grand detail') just to trigger the premium AI model. This model floods with requests and API costs rise.
* **Mitigation:** Budget Caps and Fallbacks. We implement a daily token/dollar budget for the 'advanced_model' path. If it gets breached, the router gracefully directs traffic to the 'small_model' path, while firing a Slack alert to the engineering team.

**Second Failure Mode:** *Sparse Vocab (OOD Trap)*
While the semantic router uses TF-IDF (Term Frequency - Inverse Document Frequency), it relies on exact lexical matches in the vector space. A query using completely novel vocabulary (e.g., brand new JS framework) will yield a 0 similarity score to the complex cluster, causing a fallback misroute to our cheaper and smaller model.
* **Mitigation:** In production, TF-IDF (Term Frequency - Inverse Document Frequency) vectorizer should be upgraded to a lightweight *Dense Embedding* model (e.g., 'all-MiniLM-L6-v2'). We could also implement a  *low confidence guardrail*: if 'max(sim_to_complex, sim_to_simple) < 0.15', the router flags query as strange and defaults to the advanced model to guarantee output quality, while notifying the team. To keep costs low for such a guardrail, we can place a Budget Cap on it, to ensure these anomalous requests do not get used by power users.

---

## Strategic Application: Agentic Workflows
While the goal of this implementation concerns balancing cost and latency between two LLMs, this semantic architecture can also be used to ***Multi-Agent Orchestration***.
In an agentic workflow, this TF-IDF/Cosine Similarity mechanism serves as a very low-latency
**Supervisor Agent**. Instead of routing between model sizes and capabilities, it plots user
intent in latent space to dispatch tasks dynamically to sub-agents (e.g., coding agent or
RAG retrieval agent), or to determine tool-use requirements before activating heavy reasoning models.
