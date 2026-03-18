import sqlite3
import datetime
import logging
from typing import Dict, Any
import config

logs = logging.getLogger(__name__)

def init_db() -> None:
    # ensure parent dir (e.g., /data) exists before connecting
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # context manager automatically handles committing and closing connection
    conn = sqlite3.connect(config.DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS routing_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                route TEXT NOT NULL,
                sim_simple REAL NOT NULL,
                sim_complex REAL NOT NULL,
                low_confidence INTEGER NOT NULL,
                est_cost_usd REAL NOT NULL,
                date TEXT NOT NULL
            )
        """)
        conn.commit() # save table creation
    finally:
        conn.close()
    
        
def log_decision(
    query_hash: str,
    route: str,
    sim_simple: float,
    sim_complex: float,
    low_confidence: bool,
    est_cost: float,
) -> None:
    # UTC for backend system timestamps to prevent bugs
    now = datetime.datetime.now(datetime.timezone.utc)
    ts = now.isoformat()
    date_str = now.strftime("%Y-%m-%d")
    
    conn = sqlite3.connect(config.DB_PATH)
    try:
        # parametrized queries (?, ?) prevent SQL inj attacks
        conn.execute(
            """
            INSERT INTO routing_log
            (ts, query_hash, route, sim_simple, sim_complex, low_confidence, est_cost_usd, date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, query_hash, route, sim_simple, sim_complex, int(low_confidence), est_cost, date_str)
        )
        conn.commit() # flush new row to hard drive
    finally:
        conn.close()
        
def get_metrics_today() -> Dict[str, Any]:
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    
    conn = sqlite3.connect(config.DB_PATH)
    try:
        # allows access to columns by name
        conn.row_factory = sqlite3.Row
        
        cur = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN route = 'simple' THEN 1 ELSE 0 END) as simple_count,
                SUM(CASE WHEN route = 'complex' THEN 1 ELSE 0 END) as complex_count,
                AVG(sim_simple) as avg_sim_simple,
                AVG(sim_complex) as avg_sim_complex,
                SUM(est_cost_usd) as estimated_cost_usd
            FROM routing_log
            WHERE date = ?
            """,
            (today,)
        )
        row = cur.fetchone()
        # no need for committing here as it is only read
    finally:
        conn.close()
        
        
    total = row["total"] if row["total"] else 0
    
    # zero-state handling: if no queries today return baseline config
    if total == 0:
        return {
            "total": 0.0,
            "simple_count": 0.0,
            "complex_count": 0.0,
            "avg_sim_simple": 0.0,
            "avg_sim_complex": 0.0,
            "estimated_cost_usd": 0.0,
            "budget_remaining_usd": config.BUDGET_CAP_DAILY,
        }
        
    spent = row["estimated_cost_usd"] or 0.0
    
    return {
        "total": total,
        "simple_count": row["simple_count"] or 0,
        "complex_count": row["complex_count"] or 0,
        "avg_sim_simple": row["avg_sim_simple"] or 0,
        "avg_sim_complex": row["avg_sim_complex"] or 0,
        "estimated_cost_usd": spent,
        "budget_remaining_usd": config.BUDGET_CAP_DAILY - spent
    }
    
def budget_exceeded_today() -> bool:
    metrics = get_metrics_today()
    return metrics["estimated_cost_usd"] >= config.BUDGET_CAP_DAILY

if __name__ == "__main__":
    # testing sequence
    init_db()
    print(f"DB initialized successfully at {config.DB_PATH}")
    log_decision("dummy_sha256_hash", "simple", 0.85, 0.12, False, config.COST_SIMPLE_USD)
    metrics = get_metrics_today()
    print(f"Metrics retrieved: Total Queries: {metrics['total']}, Budget Remaining: ${metrics['budget_remaining_usd']}")