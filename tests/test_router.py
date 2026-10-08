import json
from pathlib import Path

import pytest

from controlled_rag.demo import DemoLabelModel, DemoScorer, demo_corpus
from controlled_rag.routing import (FEATURE_NAMES, EvidenceRouter,
                                    LinearSoftmaxModel)


def test_router_features_six_and_missing_shortcut():
    scorer = DemoScorer()
    router = EvidenceRouter(scorer, DemoLabelModel(), 0.5)
    evidence = demo_corpus()[:1]
    result = router.route(["如何开启儿童锁", "儿童锁影响哪些车门"],
                          scorer.score_many("组合问题", evidence), evidence)
    assert len(result.features) == len(FEATURE_NAMES)
    assert result.state == "partial"
    assert result.features[4:] == (0.5, 1.0)


class OverconfidentModel:
    def predict(self, features):
        return "sufficient"


def test_false_sufficient_blocked_even_if_classifier_wrong():
    scorer = DemoScorer()
    result = EvidenceRouter(scorer, OverconfidentModel(), 0.5).route(
        ["如何开启儿童锁", "儿童锁影响哪些车门"], [0.9], demo_corpus()[:1])
    assert result.raw_model_state == "sufficient"
    assert result.state == "partial"


def test_lr_requires_real_export(tmp_path: Path):
    path = tmp_path / "router.json"
    with pytest.raises(FileNotFoundError):
        LinearSoftmaxModel.from_json(path)
    data = {"feature_names": list(FEATURE_NAMES),
            "classes": ["sufficient", "partial", "failed"],
            "coefficients": [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]],
            "intercepts": [0, 0, 0], "coverage_threshold": 0.5}
    path.write_text(json.dumps(data), encoding="utf8")
    clf, threshold = LinearSoftmaxModel.from_json(path)
    assert clf.predict((2, 1, 0, 0, 0, 0)) == "sufficient"
    assert threshold == .5
    data["feature_names"] = list(reversed(FEATURE_NAMES))
    path.write_text(json.dumps(data), encoding="utf8")
    with pytest.raises(ValueError):
        LinearSoftmaxModel.from_json(path)
