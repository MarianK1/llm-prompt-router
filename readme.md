# Serverless Cost/Latency Prompt Router for LLMs
This repository demonstrates an MLOps implementation of LLM prompt routing. It directs user queries to either a low-cost, lower-capacity model ('gpt-4o-mini') or an expensive reasoning model ('claude-3.5-sonnet').

The implementation is completely containerized, handles state management with SQLite, and is deployed to Google Cloud Run (scale-to-zero, 1Gi memory).

## Tech Stack & Infrastructure
* **Infrastructure:** Docker, Google Cloud Run (Scale-to-Zero Serverless)
* **API Framework:** FastAPI, Uvicorn
* **ML Stack:** `sentence-transformers/all-MiniLM-L6-v2`, `scikit-learn`, `numpy`
* **Data Engineering:** Pre-computed `.npy` mathematical matrices to ensure 0ms cold-start encoding latency.
* **State Management:** SQLite (with manual connection handling for high-concurrency safety).

## Architecture Progression

### V1: Heuristic Router
* **Approach:** Deterministic, taxonomy-guided selector that uses regex keywords and query length
* **Signals Used:** 
    * **Lexical:** Presence of action-oriented words (*analyze, debug*) signals the need for high reasoning, ergo a higher model. Informational verbs (*translate, summarize*) signal low reasoning, ergo the smaller model.
   * **Syntactic:** Queries over 20 words might require larger context windows and deeper attention mechanisms, which signal the route towards the higher model.
* **Tradeoffs:** Zero latency and zero compute cost. *However*, it is very brittle. 5-word math puzzles can be highly complex, but this naive heuristic might misroute it towards the less-capable model (which might not be able to solve it).

### V2: TF-IDF Centroid Router
* **Approach:** Embedding-based semantic router that maps queries into a sparse vector sspace using 'TfidfVectorizer'.
* **Signals Used:** Reliance on exact lexical matches matched to geometric proximity. Mathematical centroids are calculated (average vectors) for *"Simple"* and *"Complex"* query clusters to route new queries.
* **Tradeoffs:** Suffers from *'vector cancellation'*. Highly distinct complex concepts on average within a single centroid muddied the latent space. However, the discrete nature of the algorithm provided a safe '0.0' fallback for completely unknown vocabulary, which routes such queries immediately to the *'Complex'* cluster.

### V3: Dense-Embedding K-NN Router *(current architecture)*
*  **Approach:** Dense semantic cluster using 'sentence-transformers/all-MiniLM-L6-v2' and an in-memory 1-Nearest Neighbour (1-NN) matrix.
*  **Signals Used:** 
   * **Continuous-Space Proximity:** Queries are mapped into a 384-dimensional dense vector space. The update was made to avoid V2s *'vector cancellation'* (where averaging distinct concepts like React and Quantum Physics create a muddied centroid with no features), this architecture keeps raw embedding matrices of all baseline examples in RAM.
   * **1-Nearest Neighbour (1-NN):** When a new query arrives, its cosine similarity against every individual example is calculated, and then its 'np.max()' score is extracted to find the closest semantic match before routing.
*  **Tradeoffs:** Deeper semantic intent and context is captured. However, there is a larger memory footprint (managed through a 1Gi Google Cloud Run container to hold the model in RAM). It is also vulnerable to *continuous-space guessing* (hallucinating similarity for OOD jargon).

---

## Benchmark Results (V3 vs V2)
V3 Dense Embedding architecture was tested through evaluating it against the V2 TF-IDF baseline. 

**Evaluation Methodology:**
* **Set Dataset:** 40 blind queries spanning simple formatting tasks to highly complex CS/ML engineering questions.
* **Strict zero-lexical-overlap:** Queries were specifically designed to use out-of-domain vocabulary not present in the baseline matrices.

**Results:**
| Metric        | TF-IDF (Baseline) | Dense Embeddings (KNN) |
| :---          | :---              | :---                   |
| **Accuracy**  | **90.00%**        | 82.50%                 |
| **Precision** | 93.33%            | **100.00%**            |
| **Recall**    | **93.33%**        | 76.67%                 |
| **F1-Score**  | **0.933**         | 0.868                  |

**Conclusion:** The Dense Embedding model achieved perfect Precision, but the classical TF-IDF algorithm outperformed the Transformer model on F1-Score because its sparse vocabulary rules act as a highly effective strict-fallback mechanism for Out-of-Distribution (OOD) technical queries.

**Why TF-IDF beat a Dense-Embeddings?**
TF-IDF won because of its discrete failure mode. When confronted with highly technical, out-of-domain vocabulary (e.g., "bypass ASLR" or "Kubernetes gRPC"), TF-IDF safely output a strict `0.0` similarity. As the router defaults to the expensive model on a 0.0 score, TF-IDF effectively had a perfect safety net. 

Conversely, the dense embedding model (`all-MiniLM-L6-v2`), being small and generalized, maps *everything* into continuous space. It hallucinates similarity for niche edge-case vocabulary, resulting in middling confidence scores that accidentally cross the routing threshold and misroute complex queries to the cheaper model.

---

## **Theoretical Background and Inspiration**
The architecture of this system got inspired by recent industry research in LLM routing:
* **Taxonomy-Guided Selection:** Similar to SELECT-THEN-ROUTE (StR) framework, categorizing inputs into semantic classes before routing reduces significantly the decision space, which has been shown to reduce inference costs by up to 4 times without sacrificing accuracy.
* **Constrained Optimization:** This architecture treats routing as a problem with multiple objectives, seeking to maximize *Cost-Accuracy Frontier* by treating length and semantic proximity as proxy signals for computing cost and latency.

---

## Deployment & Safely Testing (MLOps)
Deploying a semantic router blindly can lead to massive cost spikes or degraded response quality. This system was designed for a 3-phase rollout:

1. **Offline Evaluation:** (Completed). Running strict zero-lexical-overlap scripts (`evaluate_router.py`) to benchmark continuous vs. discrete failure modes and establish F1 baselines.
2. **Shadow Routing:** Deploying the V3 router to production as a dark launch. It evaluates live traffic and logs its routing decisions to SQLite, but the actual API continues to rely on baseline heuristics. This verifies real-world memory footprint and latency constraints on Cloud Run.
3. **Canary Release:** Routing 10% of live traffic through the new mathematical model, monitoring *Average Latency (ms)* and *Cost per 1k Queries ($)* before full cutover.

---

## Failure Modes & Mitigations

**First Failure Mode:** *"Prompt Padding" Cost Spike*
Power users discover the router's semantic signals. Complex phrasing gets used to simple tasks (e.g., 'analyze this email thread in grand detail') just to trigger the premium AI model. This model floods with requests and API costs rise.
* **Mitigation:** Budget Caps and Fallbacks. We implement a daily token/dollar budget for the 'advanced_model' path. If it gets breached, the router gracefully directs traffic to the 'small_model' path, while firing a Slack alert to the engineering team.

**Second Failure Mode:** *Continuous Space Trap (V3)*
As seen by our benchmarks, small dense models guess wildly on specific jargon, misdirecting complex queries. At the same time, reliance on a discrete fallback (routing all `0.0` unknown vocabulary to the expensive model) creates a massive financial vulnerability where typos, random spam, or completely irrelevant OOD queries overload the premium API.
**Next Steps (V4):**
   * **LLM Cascading & Anomaly Detection:** The goal is to balance semantic accuracy with strict FinOps constraints. To reach this, V4 will pivot away from a single-shot router to a *"Fail-Fast Cascade."* Low-confidence queries (e.g., dense `max_score < 0.3` or TF-IDF `0.0`) will be routed to the *cheap* model (`gpt-4o-mini`) first, equipped with a strict system prompt instructing it to explicitly decline if it lacks the reasoning capability. Only upon a recognized decline will the system escalate the query to `claude-3.5-sonnet`. This protects the budget from OOD spam while safely catching edge cases.

---

## Strategic Application: Agentic Workflows
While the goal of this implementation concerns balancing cost and latency between two LLMs, this semantic architecture can also be used to ***Multi-Agent Orchestration***.
In an agentic workflow, this TF-IDF/Cosine Similarity mechanism serves as a very low-latency
**Supervisor Agent**. Instead of routing between model sizes and capabilities, it plots user
intent in latent space to dispatch tasks dynamically to sub-agents (e.g., coding agent or
RAG retrieval agent), or to determine tool-use requirements before activating heavy reasoning models.
