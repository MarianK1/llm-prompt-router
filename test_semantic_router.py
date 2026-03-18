import pytest
from semantic_router import SemanticRouter

@pytest.fixture(scope="module")
def router() -> SemanticRouter:
    # fixtures make sure we only load model once per test suite run into memory
    return SemanticRouter(simple, complex)

def test_complex_query_routing(router: SemanticRouter) -> None:
    label, sim_s, sim_c = router.route("How do I fix a memory leak in C++?")
    assert label == "complex"
    assert sim_c > 0.4
    assert sim_s < 0.3
    
def test_single_word_query(router: SemanticRouter) -> None:
    label, sim_s, sim_c = router.route("Hi!")
    assert isinstance(label, str)
    assert label in ["simple", "complex"]
    
def test_nonsense_string_low_confidence(router: SemanticRouter) -> None:
    # string with no semantic mapping should trigger low confidence fallback
    label, sim_s, sim_c = router.route("zzzzzzzqqqqqq xkscs blargh")
    assert sim_s < 0.25
    assert sim_c < 0.25