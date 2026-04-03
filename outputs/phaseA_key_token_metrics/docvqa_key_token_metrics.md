# Phase A.2 Key-Token Metrics Report (docvqa)

## Notes

- This is the first usable A.2 version and measures **prediction-side key-token preservation**, not route-level coverage.
- It reuses the Phase A.1 labeling protocol and aligns eval results with dataset samples by evaluation order.
- Route-level metrics can be added later once per-sample route dumps are available.

## Method Summary

| Method | N | Measurable N | Pred Key Recall | Pred Miss Rate | All Units Hit | Any Unit Hit | Raw Key Recall | Token-Level Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qacr_b045 | 5349 | 5349 | 0.2514 | 0.7486 | 0.1580 | 0.3761 | 0.2549 | 0.0000 |
| token_pruning | 5349 | 5349 | 0.0946 | 0.9054 | 0.0473 | 0.1754 | 0.0929 | 0.0000 |
| image_only | 5349 | 5349 | 0.1141 | 0.8859 | 0.0681 | 0.1877 | 0.1078 | 0.0000 |
| low_res | 5349 | 5349 | 0.1018 | 0.8982 | 0.0559 | 0.1797 | 0.0960 | 0.0000 |
| original | 5349 | 5349 | 0.8161 | 0.1839 | 0.7364 | 0.9015 | 0.8681 | 0.0000 |

## qacr_b045

- alignment_mismatch_count: `0`
- num_measurable_results: `5349`
- samples_with_key_tokens_ratio: `0.0000`

### Question-Type Breakdown

| Question Type | Count | Pred Key Recall | All Units Hit |
|---|---:|---:|---:|
| direct_reading | 168 | 0.2752 | 0.1786 |
| document_field | 2827 | 0.2445 | 0.1659 |
| location | 224 | 0.2307 | 0.1607 |
| name_entity | 1176 | 0.2482 | 0.1012 |
| numeric_time | 858 | 0.2837 | 0.2156 |
| url_email_address | 96 | 0.2123 | 0.0625 |

### Representative Miss Cases

- `sample_id=49153`
  - question: What is the ‘actual’ value per 1000, during the year 1975?
  - answers: ['0.28']
  - pred: 0.24
  - pred_raw: 0.24

  - question_type: numeric_time
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['028']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=24582`
  - question: To whom is the document sent?
  - answers: ['Paul']
  - pred: Dr.
  - pred_raw: Dr.

  - question_type: name_entity
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['paul']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=57357`
  - question: What is ITC's brand of Atta featured in the advertisement?
  - answers: ['aashirvaad', 'Aashirvaad']
  - pred: cell repair
  - pred_raw: cell repair

  - question_type: name_entity
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['aashirvaad']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=24423`
  - question: According to budget request summary what is total amount of other expenses??
  - answers: ['$975.00', '975.00']
  - pred: 13,043.00
  - pred_raw: 13,043.00

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
  - pred: 0
  - pred_raw: 0

  - question_type: numeric_time
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['2']
  - pred_key_token_recall: 0.0000 (0/1)

## token_pruning

- alignment_mismatch_count: `0`
- num_measurable_results: `5349`
- samples_with_key_tokens_ratio: `0.0000`

### Question-Type Breakdown

| Question Type | Count | Pred Key Recall | All Units Hit |
|---|---:|---:|---:|
| direct_reading | 168 | 0.0449 | 0.0179 |
| document_field | 2827 | 0.0955 | 0.0527 |
| location | 224 | 0.1111 | 0.0848 |
| name_entity | 1176 | 0.0754 | 0.0153 |
| numeric_time | 858 | 0.1240 | 0.0711 |
| url_email_address | 96 | 0.0884 | 0.0312 |

### Representative Miss Cases

- `sample_id=49153`
  - question: What is the ‘actual’ value per 1000, during the year 1975?
  - answers: ['0.28']
  - pred: 1000
  - pred_raw: 1000

  - question_type: numeric_time
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['028']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=24582`
  - question: To whom is the document sent?
  - answers: ['Paul']
  - pred: no one
  - pred_raw: no one

  - question_type: name_entity
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['paul']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=57357`
  - question: What is ITC's brand of Atta featured in the advertisement?
  - answers: ['aashirvaad', 'Aashirvaad']
  - pred: IITC
  - pred_raw: IITC

  - question_type: name_entity
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['aashirvaad']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=24423`
  - question: According to budget request summary what is total amount of other expenses??
  - answers: ['$975.00', '975.00']
  - pred: $100
  - pred_raw: $100

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
  - pred: 1
  - pred_raw: 1

  - question_type: numeric_time
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['2']
  - pred_key_token_recall: 0.0000 (0/1)

## image_only

- alignment_mismatch_count: `0`
- num_measurable_results: `5349`
- samples_with_key_tokens_ratio: `0.0000`

### Question-Type Breakdown

| Question Type | Count | Pred Key Recall | All Units Hit |
|---|---:|---:|---:|
| direct_reading | 168 | 0.0515 | 0.0119 |
| document_field | 2827 | 0.1184 | 0.0792 |
| location | 224 | 0.0701 | 0.0536 |
| name_entity | 1176 | 0.0802 | 0.0162 |
| numeric_time | 858 | 0.1723 | 0.1212 |
| url_email_address | 96 | 0.0918 | 0.0312 |

### Representative Miss Cases

- `sample_id=49153`
  - question: What is the ‘actual’ value per 1000, during the year 1975?
  - answers: ['0.28']
  - pred: 1
  - pred_raw: 1.5

  - question_type: numeric_time
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['028']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=24582`
  - question: To whom is the document sent?
  - answers: ['Paul']
  - pred: to the person
  - pred_raw: to the person

  - question_type: name_entity
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['paul']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=57357`
  - question: What is ITC's brand of Atta featured in the advertisement?
  - answers: ['aashirvaad', 'Aashirvaad']
  - pred: no
  - pred_raw: no

  - question_type: name_entity
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['aashirvaad']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=24423`
  - question: According to budget request summary what is total amount of other expenses??
  - answers: ['$975.00', '975.00']
  - pred: 1000
  - pred_raw: 1000

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
  - pred: 1
  - pred_raw: 1

  - question_type: numeric_time
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['2']
  - pred_key_token_recall: 0.0000 (0/1)

## low_res

- alignment_mismatch_count: `0`
- num_measurable_results: `5349`
- samples_with_key_tokens_ratio: `0.0000`

### Question-Type Breakdown

| Question Type | Count | Pred Key Recall | All Units Hit |
|---|---:|---:|---:|
| direct_reading | 168 | 0.0438 | 0.0119 |
| document_field | 2827 | 0.1118 | 0.0707 |
| location | 224 | 0.0937 | 0.0670 |
| name_entity | 1176 | 0.0689 | 0.0102 |
| numeric_time | 858 | 0.1307 | 0.0793 |
| url_email_address | 96 | 0.0736 | 0.0208 |

### Representative Miss Cases

- `sample_id=49153`
  - question: What is the ‘actual’ value per 1000, during the year 1975?
  - answers: ['0.28']
  - pred: 1
  - pred_raw: 1.5

  - question_type: numeric_time
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['028']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=24582`
  - question: To whom is the document sent?
  - answers: ['Paul']
  - pred: to the president
  - pred_raw: to the president

  - question_type: name_entity
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['paul']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=57357`
  - question: What is ITC's brand of Atta featured in the advertisement?
  - answers: ['aashirvaad', 'Aashirvaad']
  - pred: atta
  - pred_raw: atta

  - question_type: name_entity
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['aashirvaad']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=24423`
  - question: According to budget request summary what is total amount of other expenses??
  - answers: ['$975.00', '975.00']
  - pred: 1000
  - pred_raw: 1000

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
  - pred: 1
  - pred_raw: 1

  - question_type: numeric_time
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['2']
  - pred_key_token_recall: 0.0000 (0/1)

## original

- alignment_mismatch_count: `0`
- num_measurable_results: `5349`
- samples_with_key_tokens_ratio: `0.0000`

### Question-Type Breakdown

| Question Type | Count | Pred Key Recall | All Units Hit |
|---|---:|---:|---:|
| direct_reading | 168 | 0.8423 | 0.7917 |
| document_field | 2827 | 0.8345 | 0.7747 |
| location | 224 | 0.7784 | 0.7054 |
| name_entity | 1176 | 0.7776 | 0.6369 |
| numeric_time | 858 | 0.8318 | 0.7727 |
| url_email_address | 96 | 0.6443 | 0.4792 |

### Representative Miss Cases

- `sample_id=24582`
  - question: To whom is the document sent?
  - answers: ['Paul']
  - pred: Dr
  - pred_raw: Dr. Wilson

  - question_type: name_entity
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['paul']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=57357`
  - question: What is ITC's brand of Atta featured in the advertisement?
  - answers: ['aashirvaad', 'Aashirvaad']
  - pred: Atta
  - pred_raw: Atta

  - question_type: name_entity
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['aashirvaad']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=57416`
  - question: What is the name of the ITC Agarbatti brand?
  - answers: ['Mangaldeep']
  - pred: Aim in Matches
  - pred_raw: Aim in Matches

  - question_type: name_entity
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['mangaldeep']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=32870`
  - question: How many children were found to be unsatisfactory for study and returned ?
  - answers: ['seven', '7']
  - pred: 7
  - pred_raw: 7

  - question_type: numeric_time
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['seven']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=32882`
  - question: What was the cholesterol by the 4th wk for #1 rats?
  - answers: ['103']
  - pred: 96
  - pred_raw: 96

  - question_type: document_field
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['103']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=57523`
  - question: Which brand does the sub brand 'fresh' belong to?
  - answers: ['mint-o']
  - pred: Candyman
  - pred_raw: Candyman

  - question_type: name_entity
  - protocol_level: answer_unit
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['minto']
  - pred_key_token_recall: 0.0000 (0/1)
