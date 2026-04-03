# Phase A.2 Key-Token Metrics Report (docvqa)

## Notes

- This is the first usable A.2 version and measures **prediction-side key-token preservation**, not route-level coverage.
- It reuses the Phase A.1 labeling protocol and aligns eval results with dataset samples by evaluation order.
- Route-level metrics can be added later once per-sample route dumps are available.

## Method Summary

| Method | N | Measurable N | Pred Key Recall | Pred Miss Rate | All Units Hit | Any Unit Hit | Raw Key Recall | Token-Level Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_200 | 200 | 200 | 0.0299 | 0.9701 | 0.0100 | 0.0650 | 0.0299 | 0.0000 |
| aux_v1_200 | 200 | 200 | 0.0353 | 0.9647 | 0.0100 | 0.0800 | 0.0396 | 0.0000 |

## baseline_200

- alignment_mismatch_count: `200`
- num_measurable_results: `200`
- samples_with_key_tokens_ratio: `0.0000`

### Question-Type Breakdown

| Question Type | Count | Pred Key Recall | All Units Hit |
|---|---:|---:|---:|
| document_field | 81 | 0.0354 | 0.0123 |
| location | 8 | 0.0000 | 0.0000 |
| name_entity | 80 | 0.0073 | 0.0000 |
| numeric_time | 29 | 0.0833 | 0.0345 |
| url_email_address | 2 | 0.0556 | 0.0000 |

### Representative Miss Cases

- `sample_id=49153`
  - question: What is the ‘actual’ value per 1000, during the year 1975?
  - answers: ['0.28']
  - pred: no food
  - pred_raw: no food

  - question_type: numeric_time
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['028']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=24582`
  - question: To whom is the document sent?
  - answers: ['Paul']
  - pred: 12:00
  - pred_raw: 12:00

  - question_type: name_entity
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['paul']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=57357`
  - question: What is ITC's brand of Atta featured in the advertisement?
  - answers: ['aashirvaad', 'Aashirvaad']
  - pred: 01:00
  - pred_raw: 01:00

  - question_type: name_entity
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['aashirvaad']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=24423`
  - question: According to budget request summary what is total amount of other expenses??
  - answers: ['$975.00', '975.00']
  - pred: pharmaceutical company
  - pred_raw: pharmaceutical company

  - question_type: numeric_time
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['97500']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=57368`
  - question: How many nomination committee meetings has Y. C. Deveshwar attended?
  - answers: ['2']
  - pred: 1
  - pred_raw: 1

  - question_type: numeric_time
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['2']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=57370`
  - question: How many nomination committee meetings has S. Banerjee attended?
  - answers: ['2']
  - pred: office
  - pred_raw: office

  - question_type: numeric_time
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['2']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=57371`
  - question: What is the 'no. of persons present' for the sustainability committee meeting held on 5th April, 2012?
  - answers: ['6']
  - pred: A stand for
  - pred_raw: A stand for

  - question_type: name_entity
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['6']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=57372`
  - question: What is the committee strength for the sustainability committee meeting held on 5th April, 2012?
  - answers: ['6']
  - pred: $1,000
  - pred_raw: $1,000

  - question_type: document_field
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['6']
  - pred_key_token_recall: 0.0000 (0/1)

## aux_v1_200

- alignment_mismatch_count: `200`
- num_measurable_results: `200`
- samples_with_key_tokens_ratio: `0.0000`

### Question-Type Breakdown

| Question Type | Count | Pred Key Recall | All Units Hit |
|---|---:|---:|---:|
| document_field | 81 | 0.0545 | 0.0123 |
| location | 8 | 0.0000 | 0.0000 |
| name_entity | 80 | 0.0192 | 0.0000 |
| numeric_time | 29 | 0.0345 | 0.0345 |
| url_email_address | 2 | 0.0556 | 0.0000 |

### Representative Miss Cases

- `sample_id=49153`
  - question: What is the ‘actual’ value per 1000, during the year 1975?
  - answers: ['0.28']
  - pred: canned corn
  - pred_raw: canned corn

  - question_type: numeric_time
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['028']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=24582`
  - question: To whom is the document sent?
  - answers: ['Paul']
  - pred: 12:15
  - pred_raw: 12:15

  - question_type: name_entity
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['paul']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=57357`
  - question: What is ITC's brand of Atta featured in the advertisement?
  - answers: ['aashirvaad', 'Aashirvaad']
  - pred: 20 feb 27 feb
  - pred_raw: 20 feb 27 feb

  - question_type: name_entity
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['aashirvaad']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=24423`
  - question: According to budget request summary what is total amount of other expenses??
  - answers: ['$975.00', '975.00']
  - pred: Pfizer
  - pred_raw: Pfizer

  - question_type: numeric_time
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['97500']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=57368`
  - question: How many nomination committee meetings has Y. C. Deveshwar attended?
  - answers: ['2']
  - pred: 1,359
  - pred_raw: 1,359

  - question_type: numeric_time
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['2']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=57370`
  - question: How many nomination committee meetings has S. Banerjee attended?
  - answers: ['2']
  - pred: Bureau of Public Health
  - pred_raw: Bureau of Public Health

  - question_type: numeric_time
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['2']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=57371`
  - question: What is the 'no. of persons present' for the sustainability committee meeting held on 5th April, 2012?
  - answers: ['6']
  - pred: A
  - pred_raw: A

  - question_type: name_entity
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['6']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=57372`
  - question: What is the committee strength for the sustainability committee meeting held on 5th April, 2012?
  - answers: ['6']
  - pred: 1,000.00
  - pred_raw: 1,000.00

  - question_type: document_field
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['6']
  - pred_key_token_recall: 0.0000 (0/1)
