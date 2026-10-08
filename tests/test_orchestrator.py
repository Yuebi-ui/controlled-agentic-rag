import pytest

from controlled_rag.demo import DemoGenerator, build_demo_pipeline, demo_request
from controlled_rag.orchestrator import PipelineConfig, merge_round2


def test_round1_sufficient_skips_round2():
    gen = DemoGenerator()
    result = build_demo_pipeline(gen).run("如何开启儿童锁", demo_request())
    assert result.final_status == "answered"
    assert result.retrieval_rounds == 1
    assert result.route1 == "sufficient"
    assert result.route2 is None
    assert gen.calls == 1
    assert [x.evidence_id for x in result.citations] == ["E1"]


def test_partial_round2_rescued_with_original_requirements():
    gen = DemoGenerator()
    pipeline = build_demo_pipeline(gen)
    result = pipeline.run("如何开启儿童锁，以及儿童锁影响哪些车门？", demo_request())
    assert result.final_status == "answered"
    assert result.retrieval_rounds == 2
    assert (result.route1, result.route2) == ("partial", "sufficient")
    assert gen.calls == 1
    assert len(result.citations) == 2


def test_failed_round2_abstains_without_model_call():
    gen = DemoGenerator()
    result = build_demo_pipeline(gen).run("手册里有没有火箭发动机？", demo_request())
    assert result.final_status == "abstained"
    assert result.retrieval_rounds == 2
    assert (result.route1, result.route2) == ("failed", "failed")
    assert result.citations == []
    assert gen.calls == 0


def test_round2_merge_keeps_covered_r1():
    from controlled_rag.demo import demo_corpus
    a, b, c = demo_corpus()
    result = merge_round2([a, b], [c, a], cap=2, mandatory_keys=[b.evidence_key])
    assert [x.evidence_key for x in result] == [b.evidence_key, a.evidence_key]


def test_round3_cannot_be_configured():
    with pytest.raises(ValueError):
        PipelineConfig(max_retrieval_rounds=3)


def test_original_query_used_for_final_rerank_and_gate():
    pipeline = build_demo_pipeline()
    calls = []
    base = pipeline.scorer.score_many

    def spy(q, ev):
        calls.append(q)
        return base(q, ev)

    pipeline.scorer.score_many = spy
    query = "如何开启儿童锁，以及儿童锁影响哪些车门？"
    result = pipeline.run(query, demo_request())
    assert result.final_status == "answered"
    assert calls.count(query) >= 2  # both rounds rerank the ORIGINAL query
    assert calls.count("如何开启儿童锁") == 2  # original atomic requirements gate twice
    assert calls.count("儿童锁影响哪些车门") == 2


def test_empty_question():
    generator = DemoGenerator()
    result = build_demo_pipeline(generator).run("  ", demo_request())
    assert result.final_status == "abstained"
    assert result.retrieval_rounds == 0
    assert generator.calls == 0


def test_bad_model_json_fails_closed():
    class InvalidGenerator:
        def generate(self, original_query, requirements, final):
            return "```invalid json```"

    result = build_demo_pipeline(InvalidGenerator()).run("如何开启儿童锁", demo_request())
    assert result.final_status == "failed"
    assert result.citations == []
    assert result.reason_code == "MODEL_OUTPUT_OR_CITATION_INVALID"


def test_model_output_fabricated_citation_fails_closed():
    class ForgingGenerator:
        def generate(self, original_query, requirements, final):
            return '{"abstain":false,"abstain_reason":null,"answer_blocks":[{"text":"答案","evidence_ids":["E99"]}]}'

    result = build_demo_pipeline(ForgingGenerator()).run("如何开启儿童锁", demo_request())
    assert result.final_status == "failed"
    assert result.citations == []


def test_downstream_model_unavailable_fails_closed():
    class FailingGenerator:
        def generate(self, original_query, requirements, final):
            raise TimeoutError("provider timeout")

    result = build_demo_pipeline(FailingGenerator()).run("如何开启儿童锁", demo_request())
    assert result.final_status == "failed"
    assert result.reason_code == "GENERATOR_UNAVAILABLE"


def test_invalid_decomposer_contract_rejected():
    pipeline = build_demo_pipeline()
    pipeline.decomposer.decompose = lambda question: "a single string"
    result = pipeline.run("如何开启儿童锁", demo_request())
    assert result.final_status == "failed"
    assert result.retrieval_rounds == 0
