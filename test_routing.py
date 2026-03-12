from router import QueryRouter
import config

def run_tests():
    print("Running query tests for QueryRouter")
    router = QueryRouter()
    
    # test 1: simple intent
    res1 = router.get_route("What is the capital of France?")
    assert res1["model"] == config.models["small"]["name"], "Test 1 Failed!"
    
    # test 2: complex intent
    res2 = router.get_route("Debug this Python code and explain why there is a memory leak.")
    assert res2["model"] == config.models["advanced"]["name"], "Test 2 Failed!"
    
    # test 3: combined intent (length + simple keywords)
    long_simple_query = "Please summarize this very long email thread. I need to know about the upcoming meeting tomorrow at noon. Tell me what the main takeaways are so that I can prepare"
    res3 = router.get_route(long_simple_query)
    assert res3["model"] == config.models["small"]["name"], "Test 3 Failed!"
    
    print("All Unit Tests passed successfully!")
    
if __name__ == "__main__":
    run_tests()