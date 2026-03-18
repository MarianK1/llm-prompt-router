import hashlib
import logging
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from starlette.testclient import TestClient

import config
import db
from semantic_router import SemanticRouter

log = logging.getLogger(__name__)

# module level initializing to evade high latency through initing for every POST
# request => legacy
simple_examples = [
    "Translate 'hello' to French.",
    "What is the capital city of Albania?",
    "Summarize this email in one sentence: Hi all, The general meeting will be in the auditorium. Please attend and bring your thinking caps on! Kind regards, Marian",
    "What time is it in Mexico City now?",
    "Give me a quick recipe for pancakes."
]

complex_examples = [
    "Debug this Python script and explain in detail why it throws this memory error.",
    "Analyze the logic behind quantum computing and explain it to a high-schooler in a few words.",
    "Write a detailed architectural plan for migrating a Postgres database to AWS.",
    "Why did the Roman Empire fall? Compare the economic and military factors.",
    "Write a React component that manages state for a multi-step checkout form.",
]

router = SemanticRouter()

app = FastAPI(title="LLM Prompt Router", version="2.0.0")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui() -> str:
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LLM Prompt Router</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #0d1117; color: #c9d1d9; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
            .container { background-color: #161b22; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); width: 100%; max-width: 500px; border: 1px solid #30363d; }
            h2 { margin-top: 0; color: #58a6ff; }
            textarea { width: 100%; height: 100px; background-color: #0d1117; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 10px; font-size: 14px; resize: none; box-sizing: border-box; margin-bottom: 1rem; }
            textarea:focus { outline: none; border-color: #58a6ff; }
            button { background-color: #238636; color: white; border: none; padding: 10px 16px; border-radius: 6px; font-weight: 600; cursor: pointer; width: 100%; transition: background-color 0.2s; }
            button:hover { background-color: #2ea043; }
            #result { margin-top: 1.5rem; padding: 1rem; background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; display: none; font-family: monospace; white-space: pre-wrap; }
            .badge { display: inline-block; padding: 2px 6px; border-radius: 10px; font-size: 12px; font-weight: bold; margin-bottom: 8px; }
            .bg-simple { background-color: #1f6feb; color: white; }
            .bg-complex { background-color: #a371f7; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Semantic Prompt Router</h2>
            <p style="font-size: 14px; color: #8b949e; margin-bottom: 1rem;">Type a prompt to see if it requires an expensive high-reasoning LLM or a cheap, fast model.</p>
            <textarea id="query" placeholder="e.g., Explain the performance tradeoffs in a microservice cascade..."></textarea>
            <button onclick="routeQuery()">Analyze & Route</button>
            <div id="result"></div>
        </div>

        <script>
            async function routeQuery() {
                const query = document.getElementById('query').value;
                const resultDiv = document.getElementById('result');
                const btn = document.querySelector('button');
                
                if (!query) return;
                
                btn.innerText = "Analyzing...";
                resultDiv.style.display = "none";

                try {
                    const response = await fetch('/route', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: query })
                    });
                    
                    if (response.status === 429) {
                        resultDiv.innerHTML = "<span style='color: #f85149;'>Error: Daily budget cap exceeded.</span>";
                    } else if (response.status === 422) {
                        resultDiv.innerHTML = "<span style='color: #f85149;'>Error: Input must be between 1 and 2000 characters.</span>";
                    } else {
                        const data = await response.json();
                        const badgeClass = data.route === 'simple' ? 'bg-simple' : 'bg-complex';
                        const modelName = data.route === 'simple' ? 'gpt-4o-mini' : 'claude-3.5-sonnet';
                        
                        resultDiv.innerHTML = `
<span class="badge ${badgeClass}">${data.route.toUpperCase()} ROUTE</span>
Model Assigned: ${modelName}
Latency:        ${data.latency_ms} ms
Query Cost:     $${data.cost_usd}
Sim to Simple:  ${data.sim_simple}
Sim to Complex: ${data.sim_complex}
Budget Left:    $${data.budget_remaining_usd.toFixed(4)}
                        `;
                    }
                    resultDiv.style.display = "block";
                } catch (error) {
                    resultDiv.innerHTML = "<span style='color: #f85149;'>Network error occurred.</span>";
                    resultDiv.style.display = "block";
                } finally {
                    btn.innerText = "Analyze & Route";
                }
            }
        </script>
    </body>
    </html>
"""


@app.on_event("startup")
async def startup() -> None:
    # ensure db schema readiness before accepting traffic
    db.init_db()
    
class RouteRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    
class RouteResponse(BaseModel):
    route: str
    sim_simple: float
    sim_complex: float
    low_confidence: bool
    budget_remaining_usd: float
    cost_usd: float
    latency_ms: float
    
@app.post("/route", response_model=RouteResponse)
async def route_query(req: RouteRequest) -> RouteResponse:
    # fail fast if budget is spent
    if db.budget_exceeded_today():
        raise HTTPException(status_code=429, detail="Daily Budget Cap Exceeded")
    # start timer
    start_time = time.perf_counter()
    
    # route
    label, sim_s, sim_c = router.route(req.query)
    
    # stop timer
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    
    # evaluate
    low_conf = max(sim_s, sim_c) < config.LOW_CONFIDENCE_THRESHOLD
    cost = config.COST_SIMPLE_USD if label == "simple" else config.COST_ADV_USD
    
    # hashing before storing for privacy
    q_hash = hashlib.sha256(req.query.encode()).hexdigest()
    
    db.log_decision(q_hash, label, sim_s, sim_c, low_conf, cost)
    metrics = db.get_metrics_today()
    
    # presentation (float to 4 decimal places)
    return RouteResponse(
        route = label,
        sim_simple = round(sim_s, 4),
        sim_complex = round(sim_c, 4),
        low_confidence = low_conf,
        budget_remaining_usd = round(metrics["budget_remaining_usd"], 4),
        cost_usd = cost,
        latency_ms = round(latency_ms, 2)
    )
    
@app.get("/metrics")
async def get_metrics() -> dict:
    return db.get_metrics_today()

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": config.EMBEDDING_MODEL, "version": "2.0.0"}


if __name__ == "__main__":
    # testing without binding to network port
    print("Running api.py test...")
    db.init_db()
    client = TestClient(app)
    res = client.get("/health")
    print(f"API Check: {res.status_code} | {res.json()}")