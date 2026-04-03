"""Key-token labeling utilities for OCR-sensitive QACR analysis."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any


_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[.:/-][A-Za-z0-9]+)*")
_TIMEISH_RE = re.compile(r"\d{1,2}[:.]\d{2}")
_DATEISH_RE = re.compile(r"\d{1,4}[/-]\d{1,2}(?:[/-]\d{1,4})?")
_URLISH_RE = re.compile(r"(?:https?://|www\.|@|\.com\b|\.org\b|\.net\b)", re.IGNORECASE)
_MONEYISH_RE = re.compile(r"[$€£¥]|\d+\.\d+|\d+%")
_LOCATION_HINTS = {
    "where",
    "state",
    "city",
    "country",
    "town",
    "located",
    "location",
    "street",
    "address",
}
_NAME_HINTS = {
    "who",
    "name",
    "called",
    "brand",
    "store",
    "company",
    "title",
    "person",
    "man",
    "woman",
}


def _collapse_ws(text: str) -> str:
    return _WS_RE.sub(" ", str(text)).strip()


def _normalize_text(text: str) -> str:
    text = _collapse_ws(text).lower()
    text = _NON_ALNUM_RE.sub("", text)
    return text


def _normalize_tokens(text: str) -> list[str]:
    return [_normalize_text(tok) for tok in _TOKEN_RE.findall(str(text)) if _normalize_text(tok)]


def _tokenize_text(text: str) -> list[str]:
    return [tok for tok in _TOKEN_RE.findall(str(text)) if tok]


def classify_question_type(question: str, dataset_name: str) -> str:
    q = _collapse_ws(question).lower()
    if any(key in q for key in ["what time", "what year", "what date", "when", "how many", "how much", "price", "amount", "total", "value"]):
        return "numeric_time"
    if _URLISH_RE.search(q) or any(key in q for key in ["website", "email", "e-mail", "url", "address"]):
        return "url_email_address"
    if any(key in q for key in _LOCATION_HINTS):
        return "location"
    if any(key in q for key in _NAME_HINTS):
        return "name_entity"
    if any(key in q for key in ["what does", "what is written", "what word", "what number", "what letters", "what text", "what is on"]):
        return "direct_reading"
    if dataset_name == "docvqa":
        return "document_field"
    return "open"


def _choose_canonical_answer(answers: list[str]) -> str:
    cleaned = [_collapse_ws(a) for a in answers if _collapse_ws(a)]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    # Prefer majority, then longer answer to preserve exact-copy detail.
    return sorted(counts.keys(), key=lambda x: (-counts[x], -len(x), x))[0]


def _span_score(
    span_tokens: list[str],
    answer_norm: str,
    answer_units: list[str],
) -> tuple[int, float]:
    span_text = " ".join(span_tokens)
    span_norm = _normalize_text(span_text)
    if not span_norm:
        return (-1, 0.0)
    if span_norm == answer_norm:
        return (100, 1.0)
    if answer_norm and answer_norm in span_norm:
        ratio = len(answer_norm) / max(len(span_norm), 1)
        return (85, ratio)
    if span_norm and span_norm in answer_norm:
        ratio = len(span_norm) / max(len(answer_norm), 1)
        return (80, ratio)

    span_units = [_normalize_text(tok) for tok in span_tokens if _normalize_text(tok)]
    overlap = len(set(span_units) & set(answer_units))
    unit_ratio = overlap / max(len(set(answer_units)), 1)
    seq_ratio = SequenceMatcher(None, span_norm, answer_norm).ratio()
    if seq_ratio >= 0.86:
        return (70, seq_ratio)
    if unit_ratio >= 0.6 and overlap > 0:
        return (60, unit_ratio)
    return (-1, 0.0)


def _find_best_ocr_spans(
    ocr_tokens: list[str],
    answers: list[str],
    max_span_len: int = 6,
) -> list[dict[str, Any]]:
    if not ocr_tokens or not answers:
        return []

    matches: list[dict[str, Any]] = []
    norm_answers = []
    for answer in answers:
        answer_clean = _collapse_ws(answer)
        answer_norm = _normalize_text(answer_clean)
        if not answer_norm:
            continue
        answer_units = _normalize_tokens(answer_clean)
        norm_answers.append((answer_clean, answer_norm, answer_units))

    for answer_clean, answer_norm, answer_units in norm_answers:
        best: dict[str, Any] | None = None
        for start in range(len(ocr_tokens)):
            for span_len in range(1, min(max_span_len, len(ocr_tokens) - start) + 1):
                end = start + span_len
                span_tokens = ocr_tokens[start:end]
                band, score = _span_score(span_tokens, answer_norm, answer_units)
                if band < 0:
                    continue
                candidate = {
                    "answer": answer_clean,
                    "token_start": start,
                    "token_end": end,
                    "tokens": span_tokens,
                    "match_band": band,
                    "score": float(score),
                }
                if best is None:
                    best = candidate
                    continue
                best_key = (best["match_band"], best["score"], -len(best["tokens"]))
                cand_key = (candidate["match_band"], candidate["score"], -len(candidate["tokens"]))
                if cand_key > best_key:
                    best = candidate
        if best is not None:
            matches.append(best)
    return matches


def _fallback_indices_from_answer_units(
    ocr_tokens: list[str],
    answer_units: list[str],
    question_type: str,
) -> list[int]:
    if not ocr_tokens:
        return []

    indices: list[int] = []
    answer_unit_set = {unit for unit in answer_units if unit}
    digit_units = {unit for unit in answer_unit_set if any(ch.isdigit() for ch in unit)}

    for idx, token in enumerate(ocr_tokens):
        norm = _normalize_text(token)
        if not norm:
            continue
        if norm in answer_unit_set:
            indices.append(idx)
            continue
        if any(SequenceMatcher(None, norm, unit).ratio() >= 0.9 for unit in answer_unit_set):
            indices.append(idx)
            continue
        if question_type == "numeric_time":
            if digit_units and any(SequenceMatcher(None, norm, unit).ratio() >= 0.75 for unit in digit_units):
                indices.append(idx)
                continue
            if _TIMEISH_RE.search(token) or _DATEISH_RE.search(token) or _MONEYISH_RE.search(token):
                indices.append(idx)
                continue
        if question_type == "url_email_address" and _URLISH_RE.search(token):
            indices.append(idx)

    return sorted(set(indices))


def annotate_key_tokens(
    question: str,
    gt_answers: list[str] | None,
    dataset_name: str,
    ocr_tokens: list[str] | None = None,
) -> dict[str, Any]:
    answers = [_collapse_ws(a) for a in (gt_answers or []) if _collapse_ws(a)]
    canonical_answer = _choose_canonical_answer(answers)
    question_type = classify_question_type(question, dataset_name)
    answer_units = _normalize_tokens(canonical_answer)
    ocr_tokens = [str(tok) for tok in (ocr_tokens or []) if _collapse_ws(tok)]

    annotation: dict[str, Any] = {
        "dataset": dataset_name,
        "question_type": question_type,
        "canonical_answer": canonical_answer,
        "answer_units": answer_units,
        "protocol_level": "answer_unit",
        "match_strategy": "answer_units_only",
        "key_token_indices": [],
        "key_tokens": [],
        "matched_spans": [],
        "notes": [],
    }

    if not canonical_answer:
        annotation["notes"].append("missing_ground_truth_answer")
        return annotation

    if not ocr_tokens:
        annotation["notes"].append("no_ocr_tokens_available")
        return annotation

    annotation["protocol_level"] = "token"
    matches = _find_best_ocr_spans(ocr_tokens, [canonical_answer])
    matched_indices: set[int] = set()
    for match in matches:
        matched_indices.update(range(match["token_start"], match["token_end"]))

    if matched_indices:
        annotation["match_strategy"] = "ocr_span_match"
        annotation["matched_spans"] = matches
    else:
        fallback = _fallback_indices_from_answer_units(ocr_tokens, answer_units, question_type)
        matched_indices.update(fallback)
        if fallback:
            annotation["match_strategy"] = "answer_unit_fallback"
        else:
            annotation["notes"].append("no_reliable_token_match")

    annotation["key_token_indices"] = sorted(matched_indices)
    annotation["key_tokens"] = [ocr_tokens[idx] for idx in annotation["key_token_indices"]]
    return annotation


def summarize_key_token_annotations(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    question_type_counts: dict[str, int] = defaultdict(int)
    strategy_counts: dict[str, int] = defaultdict(int)
    note_counts: dict[str, int] = defaultdict(int)
    total_key_tokens = 0
    token_level = 0
    with_match = 0

    for record in records:
        question_type_counts[record.get("question_type", "unknown")] += 1
        strategy_counts[record.get("match_strategy", "unknown")] += 1
        total_key_tokens += len(record.get("key_token_indices", []))
        if record.get("protocol_level") == "token":
            token_level += 1
        if record.get("key_token_indices"):
            with_match += 1
        for note in record.get("notes", []):
            note_counts[note] += 1

    return {
        "total_samples": total,
        "token_level_samples": token_level,
        "token_level_ratio": float(token_level / total) if total else 0.0,
        "samples_with_key_tokens": with_match,
        "samples_with_key_tokens_ratio": float(with_match / total) if total else 0.0,
        "avg_key_tokens_per_sample": float(total_key_tokens / total) if total else 0.0,
        "question_type_breakdown": dict(sorted(question_type_counts.items())),
        "match_strategy_breakdown": dict(sorted(strategy_counts.items())),
        "note_breakdown": dict(sorted(note_counts.items())),
    }


def _prediction_units(text: str) -> set[str]:
    return {unit for unit in _normalize_tokens(text) if unit}


def _unit_hit(pred_units: set[str], target_unit: str) -> bool:
    if not target_unit:
        return False
    if target_unit in pred_units:
        return True
    for pred_unit in pred_units:
        if pred_unit in target_unit or target_unit in pred_unit:
            return True
        if SequenceMatcher(None, pred_unit, target_unit).ratio() >= 0.9:
            return True
    return False


def compute_prediction_key_token_metrics(
    prediction: str,
    annotation: dict[str, Any],
) -> dict[str, Any]:
    pred_units = _prediction_units(prediction)
    answer_units = [unit for unit in annotation.get("answer_units", []) if unit]
    key_tokens = annotation.get("key_tokens", []) or []
    key_token_units = [_normalize_text(tok) for tok in key_tokens if _normalize_text(tok)]
    target_units = key_token_units or answer_units

    if not target_units:
        return {
            "target_unit_count": 0,
            "matched_target_units": 0,
            "key_token_recall": 0.0,
            "key_token_miss_rate": 0.0,
            "all_target_units_hit": False,
            "any_target_unit_hit": False,
            "prediction_units": sorted(pred_units),
        }

    matched = sum(1 for unit in target_units if _unit_hit(pred_units, unit))
    recall = matched / len(target_units)
    return {
        "target_unit_count": len(target_units),
        "matched_target_units": matched,
        "key_token_recall": float(recall),
        "key_token_miss_rate": float(1.0 - recall),
        "all_target_units_hit": matched == len(target_units),
        "any_target_unit_hit": matched > 0,
        "prediction_units": sorted(pred_units),
    }
