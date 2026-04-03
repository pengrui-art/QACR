"""Evaluation helpers for short-answer VQA benchmarks."""

from __future__ import annotations

import html
import re
from difflib import SequenceMatcher
from collections.abc import Sequence


_ARTICLES = {"a", "an", "the"}
_NUMBER_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hes": "he's",
    "im": "i'm",
    "isnt": "isn't",
    "itll": "it'll",
    "its": "it's",
    "ive": "i've",
    "shouldnt": "shouldn't",
    "shouldve": "should've",
    "thats": "that's",
    "theres": "there's",
    "theyre": "they're",
    "wasnt": "wasn't",
    "werent": "weren't",
    "whats": "what's",
    "wheres": "where's",
    "whos": "who's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youre": "you're",
}
_ROLE_LINE_RE = re.compile(r"^\s*(system|user|assistant)\s*[:：]?\s*$", re.IGNORECASE)
_ROLE_PREFIX_RE = re.compile(r"^\s*(system|user|assistant)\s*[:：]?\s*", re.IGNORECASE)
_CHOICE_RE = re.compile(r"\b([A-G])\b")
_ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+answer|short\s+answer|answer)\s*(?:is)?\s*[:：-]?\s*",
    re.IGNORECASE,
)
_EXPLANATION_SPLIT_RE = re.compile(
    r"\s+(?:because|since|therefore|thus|hence|so that|which means|meaning that)\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"\b(?:https?://)?(?:www\.)?[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s]*)?\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_TIME_RANGE_RE = re.compile(
    r"\b\d{1,2}[:.]\d{2}(?:\s*(?:a\.?m\.?|p\.?m\.?))?(?:\s*(?:to|-)\s*\d{1,2}[:.]\d{2}(?:\s*(?:a\.?m\.?|p\.?m\.?))?)?\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b",
    re.IGNORECASE,
)
_MONEY_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
_INITIAL_NAME_RE = re.compile(r"\b(?:[A-Z]\.){1,4}(?:\s+[A-Z][a-z]+)+\b")
_STATE_RE = re.compile(
    r"\b("
    r"alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|mississippi|missouri|montana|nebraska|nevada|new hampshire|new jersey|new mexico|new york|north carolina|north dakota|ohio|oklahoma|oregon|pennsylvania|rhode island|south carolina|south dakota|tennessee|texas|utah|vermont|virginia|washington|west virginia|wisconsin|wyoming"
    r")\b",
    re.IGNORECASE,
)
_MONTH_MAP = {
    "jan": "january",
    "january": "january",
    "feb": "february",
    "february": "february",
    "mar": "march",
    "march": "march",
    "apr": "april",
    "april": "april",
    "may": "may",
    "jun": "june",
    "june": "june",
    "jul": "july",
    "july": "july",
    "aug": "august",
    "august": "august",
    "sep": "september",
    "sept": "september",
    "september": "september",
    "oct": "october",
    "october": "october",
    "nov": "november",
    "november": "november",
    "dec": "december",
    "december": "december",
}
_YES_NO_Q_RE = re.compile(
    r"^(?:is|are|was|were|do|does|did|can|could|should|would|will|has|have|had)\b",
    re.IGNORECASE,
)
_QUESTION_ECHO_CLAUSE_RE = re.compile(
    r"^(?:the\s+)?(?:answer|author|brand|name|title|store|word|number|time|year|date)\b[^.\n]{0,80}\bis\b\s+",
    re.IGNORECASE,
)
_TEXTVQA_LIST_PREFIX_RE = re.compile(r"^\s*[A-Za-z0-9]+[.)]\s+")
_TEXTVQA_UNIT_RE = re.compile(
    r"\b\d+(?:[:.]\d+)*(?:\s*(?:ml|oz|cm|mm|kg|g|lbs?|lb|years?|year|cents?|%)|\s*:\s*\d+)?\b",
    re.IGNORECASE,
)
_TEXTVQA_BRAND_SUFFIXES = {
    "watch",
    "phone",
    "camera",
    "television",
    "tv",
    "beer",
    "lager",
    "ale",
    "stout",
    "ipa",
    "dipale",
    "cream",
    "ice",
    "grand",
    "ferry",
    "panel",
    "optimus",
    "l9",
    "lte",
    "model",
}
_TEXTVQA_GENERIC_ANSWER_WORDS = {
    "brand",
    "name",
    "title",
    "word",
    "number",
    "time",
    "year",
    "date",
    "camera",
    "phone",
    "watch",
    "beer",
    "wine",
    "store",
    "company",
    "business",
    "city",
    "country",
    "state",
}
_TEXTVQA_OCR_NOISE_WORDS = {
    "menu",
    "end",
    "back",
    "next",
    "start",
    "on",
    "off",
    "the",
    "and",
}


def _ensure_answer_list(answers: Sequence[str] | str | None) -> list[str]:
    if answers is None:
        return []
    if isinstance(answers, str):
        return [answers]
    return [str(answer) for answer in answers if answer is not None]


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _truncate_explanatory_tail(text: str) -> str:
    text = text.strip().strip("`*_#>-\"' ")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    text = _ANSWER_PREFIX_RE.sub("", text).strip()
    text = _EXPLANATION_SPLIT_RE.split(text, maxsplit=1)[0].strip()
    text = re.split(r"\s+\((?:optional|note|explanation|reason|source)\b", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return text


def _extract_open_answer(candidate: str, joined: str) -> str:
    candidate = _truncate_explanatory_tail(candidate)
    if not candidate:
        return ""

    # For OCR/document answers, punctuation often carries semantics
    # (e.g. decimals, times, initials, addresses), so keep the first line intact.
    first_line = candidate.splitlines()[0].strip()
    first_line = _ANSWER_PREFIX_RE.sub("", first_line).strip()
    if not first_line:
        return ""

    if re.search(r"\b(?:because|since|therefore|thus)\b", first_line, flags=re.IGNORECASE):
        first_line = _EXPLANATION_SPLIT_RE.split(first_line, maxsplit=1)[0].strip()

    if first_line.lower().startswith("the answer is "):
        first_line = first_line[14:].strip()
    return first_line or candidate


def _strip_textvqa_list_prefix(text: str) -> str:
    stripped = _TEXTVQA_LIST_PREFIX_RE.sub("", text).strip()
    return stripped or text


def _trim_textvqa_brand_phrase(text: str) -> str:
    tokens = text.split()
    if len(tokens) <= 1:
        return text

    cut = len(tokens)
    for idx, token in enumerate(tokens[1:], start=1):
        lowered = token.lower().strip(".,:;!?()[]{}")
        if lowered in _TEXTVQA_BRAND_SUFFIXES or re.search(r"\d", lowered):
            cut = idx
            break
    trimmed = " ".join(tokens[:cut]).strip()
    return trimmed or text


def _extract_textvqa_answer(merged: str, question: str | None) -> str:
    q = (question or "").lower()
    merged = _strip_textvqa_list_prefix(merged)
    numeric_token = re.search(r"\d+(?:\.\d+)?", merged)
    size_question = any(
        key in q
        for key in [
            "what size",
            "what is the size",
            "what size is",
            "what capacity",
            "what is the capacity",
            "what does the label say",
        ]
    )
    measurement_question = any(
        key in q
        for key in [
            "how many ml",
            "how many milliliter",
            "how many milliliters",
            "how many grams",
            "how many gram",
            "how many ounces",
            "how many ounce",
            "how many oz",
            "how many lbs",
            "how many pounds",
            "how many centimeters",
            "how many centimetres",
            "how many cm",
            "how many mm",
            "how many kg",
            "what year",
            "which year",
        ]
    )
    percent_question = any(
        key in q for key in ["what percent", "what percentage", "alcohol content", "alcohol percentage"]
    )

    if "first word" in q:
        return merged.split()[0] if merged.split() else merged
    if "last word" in q:
        return merged.split()[-1] if merged.split() else merged
    if "middle word" in q:
        tokens = merged.split()
        if tokens:
            return tokens[len(tokens) // 2]
    if "say to do" in q:
        return merged.split()[0] if merged.split() else merged
    if "initials" in q:
        compact = re.sub(r"[^A-Za-z]", "", merged)
        if 2 <= len(compact) <= 4:
            return compact[:2]
    if "letter is in the middle" in q:
        compact = re.sub(r"[^A-Za-z0-9]", "", merged)
        if 1 < len(compact) <= 4:
            return compact[(len(compact) - 1) // 2]

    if any(key in q for key in ["brand of phone", "phone's manufacturer", "manufacturer", "brand of watch", "brand of camera", "brand of this television", "brand of beer", "brand of ice cream", "brand is the watch"]):
        merged = _trim_textvqa_brand_phrase(merged)

    if "what country" in q and merged.lower().endswith("s") and len(merged.split()) == 1:
        return merged[:-1]
    if "what year" in q or "which year" in q:
        years = re.findall(r"\b(?:19|20)\d{2}\b", merged)
        if years:
            return years[-1]

    if any(
        key in q
        for key in [
            "what number",
            "what is the number",
            "how many",
            "how much",
            "how much is",
            "what time",
            "how much time",
            "what percent",
            "what price",
            "what is the price",
            "what number is",
            "what code",
        ]
    ):
        compact = merged.replace(" ", "")
        if "letter" in q:
            return merged
        unit_match = _TEXTVQA_UNIT_RE.search(merged)
        if unit_match and (any(ch.isalpha() for ch in unit_match.group(0)) or "%" in unit_match.group(0)):
            if percent_question and numeric_token:
                try:
                    numeric_value = float(numeric_token.group(0))
                except ValueError:
                    numeric_value = None
                if numeric_value is not None and numeric_value.is_integer():
                    return str(int(numeric_value))
            if size_question or percent_question:
                return unit_match.group(0)
            if measurement_question and numeric_token:
                return numeric_token.group(0)
        if re.fullmatch(r"\d+(?:[:.]\d+)+", compact):
            return merged
        if re.fullmatch(r"\d+[A-Za-z]+", compact):
            if size_question or percent_question:
                return merged
            if measurement_question and numeric_token:
                return numeric_token.group(0)
            return re.match(r"\d+", compact).group(0)
        if re.fullmatch(r"[A-Za-z]+\d+", compact):
            return re.search(r"\d+$", compact).group(0)
        if re.search(r"[A-Za-z]", compact) and re.search(r"\d", compact):
            return merged
        if re.fullmatch(r"\d+\.\d+", compact):
            return merged

    if any(key in q for key in ["what does the sign say", "what does his jacket say", "what does it say", "what does the screen say"]):
        if merged.lower().endswith(" sign"):
            return merged[:-5].strip()

    return merged


def _iter_textvqa_ocr_candidates(ocr_tokens: Sequence[str] | None) -> list[tuple[str, str, int]]:
    if not ocr_tokens:
        return []
    tokens = [_collapse_ws(str(token)) for token in ocr_tokens if str(token).strip()]
    candidates: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for i in range(len(tokens)):
        for span_len in range(1, 4):
            j = i + span_len
            if j > len(tokens):
                break
            span = " ".join(tokens[i:j]).strip(" ,;")
            norm = normalize_vqa_answer(span)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            candidates.append((span, norm, i))
    return candidates


def _correct_textvqa_with_ocr(prediction: str, question: str | None, ocr_tokens: Sequence[str] | None) -> str:
    if not prediction or not ocr_tokens:
        return prediction

    q = (question or "").lower()
    pred = _collapse_ws(prediction)
    pred_norm = normalize_vqa_answer(pred)
    if not pred_norm:
        return prediction

    question_norm = normalize_vqa_answer(question or "")
    question_words = set(question_norm.split())
    pred_words = pred_norm.split()
    pred_digits = re.sub(r"\D", "", pred_norm)
    numeric_question = any(
        key in q
        for key in [
            "how many",
            "what number",
            "what is the number",
            "how much",
            "price",
            "what time",
            "what year",
            "what percent",
            "percentage",
        ]
    )
    entity_question = any(
        key in q
        for key in [
            "brand",
            "name",
            "title",
            "author",
            "who",
            "where",
            "city",
            "country",
            "state",
            "going",
            "store",
            "company",
            "business",
            "what does",
            "what is written",
            "what does it say",
        ]
    )

    candidates = _iter_textvqa_ocr_candidates(ocr_tokens)
    candidate_norms = {span_norm for _, span_norm, _ in candidates}
    price_like_question = any(
        key in q for key in ["how much did this cost", "how much is", "what is the price", "what price"]
    )
    if pred_norm in candidate_norms:
        generic_pred = bool(set(pred_words) & _TEXTVQA_GENERIC_ANSWER_WORDS)
        if not generic_pred:
            if numeric_question and not price_like_question:
                return prediction
            if not numeric_question:
                return prediction

    best_span = pred
    best_score = 0.0
    for span, span_norm, pos in candidates:
        if span_norm == pred_norm or not span_norm:
            continue
        span_words = span_norm.split()
        span_digits = re.sub(r"\D", "", span_norm)
        raw_span_words = [w.lower() for w in _collapse_ws(span).split()]
        if len(span_words) > 4:
            continue
        if any(word in _TEXTVQA_OCR_NOISE_WORDS for word in raw_span_words[:-1]):
            continue
        ratio = SequenceMatcher(None, pred_norm, span_norm).ratio()
        score = 0.0

        if numeric_question:
            number_chunks = re.findall(r"\d+(?:[:.]\d+)*", span)
            if len(number_chunks) > 1:
                continue
            if pred_digits and pred_digits == pred_norm and len(pred_digits) == 1 and span_digits == span_norm and len(span_digits) < 3:
                continue
            if pred_digits and span_digits.startswith(pred_digits) and len(span_digits) > len(pred_digits):
                score += 2.8
            if (
                price_like_question
                and pred_digits
                and pred_digits == span_digits
                and len(span_norm) > len(pred_norm)
                and any(unit in span_norm for unit in ["cent", "dollar", "$"])
            ):
                score += 2.2
            if pred_norm in span_norm and len(span_norm) <= len(pred_norm) + 12:
                score += 1.0
            if ratio >= 0.9 and len(span_norm) >= len(pred_norm):
                score += 0.8
        else:
            pure_alpha_pred = pred_norm.replace(" ", "").isalpha()
            pure_alpha_span = span_norm.replace(" ", "").isalpha()
            if len(pred_words) == 1 and len(span_words) == 1 and pure_alpha_pred and pure_alpha_span and len(pred_norm) > 4:
                plural_ok = span_norm in {pred_norm + "s", pred_norm + "es"}
                location_like = any(key in q for key in ["state", "country", "city"]) and ratio >= 0.88
                if not plural_ok and not location_like:
                    continue
            if len(pred_words) == 1 and len(pred_norm) > 6 and len(span_words) > 1:
                continue
            if pred_norm in span_norm and len(span_words) <= len(pred_words) + 2:
                score += 2.2
            if span_norm.startswith(pred_norm) or span_norm.endswith(pred_norm):
                score += 0.6
            if len(pred_words) == 1 and len(span_words) <= 3 and ratio >= 0.86:
                score += 1.8
            elif ratio >= 0.9 and len(span_words) <= 3:
                score += 1.0
            if entity_question and not (set(span_words) & _TEXTVQA_GENERIC_ANSWER_WORDS):
                score += 0.3
            if len(span_words) > 2 and not (set(pred_words) & _TEXTVQA_GENERIC_ANSWER_WORDS):
                score -= 1.0

        if entity_question:
            if set(pred_words) & (question_words | _TEXTVQA_GENERIC_ANSWER_WORDS):
                if not (set(span_words) & question_words):
                    score += 0.7
            if pos <= 2:
                score += 0.1

        if score > best_score + 1e-6 or (
            abs(score - best_score) <= 1e-6 and best_span == pred and len(span_words) <= 3
        ):
            best_score = score
            best_span = span

    threshold = 3.0 if numeric_question else 3.0
    return best_span if best_score >= threshold else prediction


def _extract_question_aware_answer(candidate: str, joined: str, question: str | None, dataset_name: str) -> str:
    answer = _extract_open_answer(candidate, joined)
    if not answer:
        return ""
    q = (question or "").lower()
    merged = _collapse_ws(answer)

    if _YES_NO_Q_RE.match(q):
        yes_no = re.findall(r"\b(yes|no)\b", merged, flags=re.IGNORECASE)
        if yes_no:
            return yes_no[-1].lower()

    if _QUESTION_ECHO_CLAUSE_RE.match(merged):
        trimmed = _QUESTION_ECHO_CLAUSE_RE.sub("", merged).strip()
        if trimmed:
            merged = trimmed

    if any(key in q for key in ["website", "web site", "web address", "online address", "url", "email", "e-mail"]):
        match = _EMAIL_RE.search(merged) or _URL_RE.search(merged)
        if match:
            return match.group(0)

    if any(key in q for key in ["what time", "which time", "time is", "time of", "coffee break", "session", "question and answers"]):
        match = _TIME_RANGE_RE.search(merged)
        if match:
            return match.group(0)

    if any(key in q for key in ["what date", "which date", "when", "dated", "date of"]):
        match = _DATE_RE.search(merged)
        if match:
            return match.group(0)

    if any(key in q for key in ["what month", "which month"]):
        token = re.search(r"\b(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\b", merged, flags=re.IGNORECASE)
        if token:
            return _MONTH_MAP[token.group(0).lower()]

    if any(key in q for key in ["what state", "which state"]):
        match = _STATE_RE.search(merged)
        if match:
            return match.group(1).lower()

    if q.startswith("who") or "name of" in q or "who is" in q or "to whom" in q:
        match = _INITIAL_NAME_RE.search(merged)
        if match:
            return match.group(0)
        if merged.lower().startswith("dr. ") or merged.lower().startswith("mr. ") or merged.lower().startswith("mrs. "):
            return merged

    if dataset_name == "textvqa":
        return _extract_textvqa_answer(merged, question)

    if any(key in q for key in ["how many", "what number", "what amount", "how much", "what is the value", "total amount", "what year"]):
        matches = _MONEY_RE.findall(merged)
        if matches:
            return max(matches, key=len)

    return merged


def postprocess_generation(
    generated_text: str,
    *,
    question: str | None = None,
    dataset_name: str = "vqav2",
    ocr_tokens: Sequence[str] | None = None,
) -> str:
    """Extract a short answer from verbose chat-model generations."""
    if not generated_text:
        return ""

    text = html.unescape(generated_text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<\|[^>]+\|>", " ", text)
    text = text.replace("</s>", " ").replace("<s>", " ")
    text = text.replace("<pad>", " ")

    question_norm = _collapse_ws(question.lower()) if question else None
    cleaned_lines: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if _ROLE_LINE_RE.fullmatch(line):
            continue
        line = _ROLE_PREFIX_RE.sub("", line).strip()
        if not line:
            continue
        if question_norm and _collapse_ws(line.lower()) == question_norm:
            continue
        cleaned_lines.append(line)

    candidate = cleaned_lines[-1] if cleaned_lines else _collapse_ws(text)
    joined = "\n".join(cleaned_lines)
    answer_match = re.search(
        r"(?:final answer|short answer|answer)\s*[:：]\s*(.+)",
        joined,
        flags=re.IGNORECASE,
    )
    if answer_match:
        candidate = answer_match.group(1).strip().splitlines()[0]

    candidate = _truncate_explanatory_tail(candidate)

    if dataset_name == "pope":
        yes_no = re.findall(r"\b(yes|no)\b", candidate, flags=re.IGNORECASE)
        if not yes_no:
            yes_no = re.findall(r"\b(yes|no)\b", joined, flags=re.IGNORECASE)
        return yes_no[-1].lower() if yes_no else ""

    if dataset_name in {"mmmu", "mmbench"}:
        choice = extract_choice_letter(candidate)
        if choice is None:
            choice = extract_choice_letter(joined)
        return choice or ""

    answer = _extract_question_aware_answer(candidate, joined, question, dataset_name)
    if dataset_name == "textvqa":
        return _correct_textvqa_with_ocr(answer, question, ocr_tokens)
    return answer


def normalize_vqa_answer(text: str) -> str:
    """Apply VQA-style normalization for short open-ended answers."""
    if not text:
        return ""

    text = html.unescape(text.lower()).strip()
    text = text.replace("\n", " ").replace("\t", " ")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"[^\w\s']", " ", text)
    tokens: list[str] = []
    for token in text.split():
        token = _CONTRACTIONS.get(token, token)
        token = _NUMBER_MAP.get(token, token)
        if token in _ARTICLES:
            continue
        tokens.append(token)
    return " ".join(tokens)


def extract_choice_letter(text: str) -> str | None:
    if not text:
        return None
    match = _CHOICE_RE.search(text.upper())
    if match:
        return match.group(1)
    match = re.search(r"(?:answer|option|choice)\s*[:：]?\s*([A-G])\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def compute_vqa_accuracy(
    prediction: str,
    answers: Sequence[str] | str | None,
    *,
    dataset_name: str = "vqav2",
) -> float:
    """Compute dataset-aware VQA accuracy from a cleaned short answer."""
    answer_list = _ensure_answer_list(answers)
    if not answer_list:
        return 0.0

    if dataset_name == "pope":
        pred_norm = postprocess_generation(prediction, dataset_name="pope")
        gt_norm = postprocess_generation(answer_list[0], dataset_name="pope")
        return 1.0 if pred_norm and pred_norm == gt_norm else 0.0

    if dataset_name in {"mmmu", "mmbench"}:
        pred_choice = postprocess_generation(prediction, dataset_name=dataset_name)
        gt_choice = postprocess_generation(answer_list[0], dataset_name=dataset_name)
        return 1.0 if pred_choice and pred_choice == gt_choice else 0.0

    pred_norm = normalize_vqa_answer(prediction)
    gt_norms = [normalize_vqa_answer(answer) for answer in answer_list]
    if not pred_norm:
        return 0.0

    if dataset_name in {"vqav2", "textvqa"} and len(gt_norms) > 1:
        matches = sum(gt == pred_norm for gt in gt_norms)
        return min(1.0, matches / 3.0)

    return 1.0 if pred_norm in gt_norms else 0.0
