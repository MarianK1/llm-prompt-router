from router import QueryRouter

def run_eval():
    """
    Simulates First Phase Eval to measure Cost-Accuracy Frontier
    """
    
    eval_dataset = [
        # expected small
        "Translate 'hello' to French.",
        "What is the capital city of Albania?",
        "Summarize this email in one sentence: Hey all, We are meeting tomorrow at 6pm. Everyone's attendance is required.",
        "What time is it in Bogota now?",
        "Give me a quick recipe for pancakes.",
        "Hi, just testing the system.",
        "Can you reply with a random fact about space?",
        "What is 2+2?",
        
        # expected advanced
        "Debug this Python script and explain in detail why it throws this memory error.",
        "Analyze the logic behind quantum computing and explain it to a high-schooler in a few words.",
        "Write a detailed architectural plan for migrating a Postgres database to AWS.",
        "Why did the Roman Empire fall? Compare the economic and military factors.",
        "Write a React component that manages state for a multi-step checkout form.",
        "Prove that no positive integers a, b, and c satisfy the equation a^n+b^n=c^n.",
        "Explain the performance tradeoffs of using an LLM cascade over a single model."
    ]
    
    router = QueryRouter()
    advanced_model_count = 0
    small_model_count = 0
    
    print(f"\n{'Query Snippet':<50} | {'Assigned Model':<20}")
    print("-" * 50)
    
    for query in eval_dataset:
        result = router.get_route(query)
        # suppressing internal logging for summary table
        if result["cost_tier"] == "high":
            advanced_model_count += 1
        else:
            small_model_count += 1
            
        print(f"{query[:47] + '...' if len(query) > 47 else query:<50} | {result['model']:<20}")
        
    # calculating metrics
    total = len(eval_dataset)
    savings_proxy = (small_model_count / total) * 100
    
    print("\n" + "=" * 50)
    print("Phase 1: Offline Evaluation Summary")
    print("=" * 50)
    print(f"Total Queries Evaluated: {total}")
    print(f"Routed to Advanced Path: {advanced_model_count} ({(advanced_model_count/total)*100:.1f}%)")
    print(f"Routed to Small Path:    {small_model_count} ({(small_model_count/total)*100:.1f}%)")
    print("-"*50)
    print(f"Estimated Cost Reduction compared to Baseline: ~{savings_proxy:.1f}%")
    print("="*50)
    
if __name__ == "__main__":
    run_eval()
    