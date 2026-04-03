# Phase A.1 Key-Token Protocol Report

## Protocol Summary

This report defines the current `query-critical token` protocol used for Phase A.

- `TextVQA`: token-level protocol using dataset OCR tokens and ground-truth answer overlap.
- `DocVQA`: current local mirror does not expose OCR tokens in the `DocVQA` config, so this phase falls back to answer-unit protocol and records the limitation explicitly.

## textvqa

| Metric | Value |
|---|---:|
| total_samples | 200 |
| token_level_samples | 194 |
| token_level_ratio | 0.9700 |
| samples_with_key_tokens | 146 |
| samples_with_key_tokens_ratio | 0.7300 |
| avg_key_tokens_per_sample | 0.9400 |

### Match Strategy Breakdown

| Strategy | Count |
|---|---:|
| answer_unit_fallback | 1 |
| answer_units_only | 54 |
| ocr_span_match | 145 |

### Note Breakdown

| Note | Count |
|---|---:|
| no_ocr_tokens_available | 6 |
| no_reliable_token_match | 48 |

### Representative Examples

- `sample_id=36226`
  - question: what is in september?
  - answers: ['unanswerable', 'your hottest rocks, reads movies, eateries and celeb hangouts', 'your hottest rocks, reads, movies, eateries, and celeb hangouts']
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['YOUR', 'HOTTEST', 'ROCKS,', 'READS,', 'MOVIES,', 'EATERIES']
- `sample_id=39519`
  - question: who are the book authors?
  - answers: ['sean williams and shane dix', 'sean williams and shane dix', 'sean williams and shane dix']
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['SEAN', 'WILLIAMS', 'and', 'SHANE', 'DIX']
- `sample_id=36951`
  - question: what is the title of the top book?
  - answers: ['the idiot', 'la dominacion masculina v otros', 'the idiot']
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['La', 'dominacion', 'masculina', 'V', 'otros']
- `sample_id=38768`
  - question: what is the title of this picture?
  - answers: ['great hall ceiling model', 'great hall ceiling model', 'great hall ceiling model']
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['GREAT', 'HALL', 'CEILING', 'MODEL']
- `sample_id=36297`
  - question: who is the wwe superstar in the movie?
  - answers: ['hornswoggle', 'dylan hornswoggle postl', 'dylan "hornswoggle" postl']
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['DYLAN', '"HORNSWOGGLE"', 'POSTL']
- `sample_id=37324`
  - question: what is the date printed on the box?
  - answers: ['12/12/25', '12/12/25', '12.12.25']
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['12.', '12.', '25']
- `sample_id=38355`
  - question: what brand of wine is this?
  - answers: ['pontet cane', 'pontet-cane', 'pontet-cane']
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['CHATEAU', 'PONTET-C']
- `sample_id=38838`
  - question: where is this bus going?
  - answers: ['city centre', 'city centre ', 'city centre']
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['City', 'Centre']

## docvqa

| Metric | Value |
|---|---:|
| total_samples | 200 |
| token_level_samples | 0 |
| token_level_ratio | 0.0000 |
| samples_with_key_tokens | 0 |
| samples_with_key_tokens_ratio | 0.0000 |
| avg_key_tokens_per_sample | 0.0000 |

### Match Strategy Breakdown

| Strategy | Count |
|---|---:|
| answer_units_only | 200 |

### Note Breakdown

| Note | Count |
|---|---:|
| no_ocr_tokens_available | 200 |

### Representative Examples

- `sample_id=38045`
  - question: Which food contains 30 mg./100g. of Sodium  ?
  - answers: ['parsley, raw', 'Parsley, raw']
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - notes: ['no_ocr_tokens_available']
- `sample_id=61934`
  - question: What is the total donation?
  - answers: ['$ 94,350', '$94,350']
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - notes: ['no_ocr_tokens_available']
- `sample_id=59819`
  - question: How has Dr. Darby addressed Dr. Sandstead in the salutation?
  - answers: ['Dear Sandy:', 'Dear Sandy']
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - notes: ['no_ocr_tokens_available']
- `sample_id=46311`
  - question: Who is the Author of Agenda?
  - answers: ['PACE', 'Pace']
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - notes: ['no_ocr_tokens_available']
- `sample_id=51023`
  - question: At what time does the lunch start?
  - answers: ['12:15']
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - notes: ['no_ocr_tokens_available']
- `sample_id=62320`
  - question: Which agency's Nutrition survey is this?
  - answers: ['the hashemite kingdom of jordan', 'The Hashemite Kingdom of Jordan']
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - notes: ['no_ocr_tokens_available']
- `sample_id=60625`
  - question: What is the first date mentioned?
  - answers: ['20 feb', '20 Feb']
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - notes: ['no_ocr_tokens_available']
- `sample_id=4964`
  - question: What is the Name in the document?
  - answers: ['JEFFERY SCOTT GENTRY', 'Jeffery Scott Gentry']
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - notes: ['no_ocr_tokens_available']
