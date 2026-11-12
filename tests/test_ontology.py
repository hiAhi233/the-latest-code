"""否定作用域：句内、后置否定、对比连词重置。"""

from __future__ import annotations

from medrrg.ontology import Ontology

ONTOLOGY = Ontology(name="negation", concepts={
    "edema": ["edema", "oedema"],
    "glioma": ["glioma", "glial tumor"],
    "enhancement": ["enhancement", "enhancing lesion"],
    "hemorrhage": ["hemorrhage", "bleeding"],
    "mass_effect": ["mass effect"],
})


def test_negation_does_not_leak_across_sentences():
    # 旧实现会用 40 字符前缀把下一句的 glioma 也判成否定
    assert ONTOLOGY.match("No edema. A glioma is present.") == ["glioma"]


def test_negation_does_not_leak_across_semicolon():
    assert ONTOLOGY.match("No edema; a glioma is present.") == ["glioma"]


def test_postposed_negation_is_detected():
    assert ONTOLOGY.match("Edema is absent.") == []
    assert ONTOLOGY.match("No enhancement was observed.") == []
    assert ONTOLOGY.match("Hemorrhage was not seen.") == []


def test_postposed_negation_does_not_cross_comma_clause():
    # 旧实现按字符向后搜索，会把 edema 后面的 "absent" 当成 edema 的否定
    assert ONTOLOGY.match("Edema is present, enhancement is absent.") == ["edema"]
    assert ONTOLOGY.match("Glioma is seen, edema is absent.") == ["glioma"]


def test_postposed_negation_does_not_cross_coordinated_clause():
    assert ONTOLOGY.match("Edema is present and enhancement is absent.") == ["edema"]


def test_postposed_negation_does_not_cross_prefix_clause():
    assert ONTOLOGY.match("Edema is present, no enhancement is seen.") == ["edema"]


def test_coordinated_subjects_share_absent_predicate():
    assert ONTOLOGY.match("Edema and hemorrhage are absent.") == []


def test_list_negation_with_commas_stays_negated():
    assert ONTOLOGY.match("No edema, hemorrhage, or mass effect.") == []


def test_contrastive_conjunction_resets_prefix_scope():
    assert ONTOLOGY.match("No edema but a glioma is present.") == ["glioma"]


def test_positive_mentions_stay_positive():
    assert ONTOLOGY.match("A glioma with surrounding edema.") == ["edema", "glioma"]


def test_include_negated_returns_all_mentions():
    assert set(ONTOLOGY.match("No edema. A glioma is present.",
                              include_negated=True)) == {"edema", "glioma"}


def test_decimal_point_does_not_split_sentence():
    assert ONTOLOGY.match("No edema measuring 2.5 cm. A glioma is present.") == ["glioma"]


def test_chinese_negation():
    ontology = Ontology(name="zh", concepts={"edema": ["水肿"], "glioma": ["胶质瘤"]})
    assert ontology.match("未见水肿。胶质瘤。") == ["glioma"]
