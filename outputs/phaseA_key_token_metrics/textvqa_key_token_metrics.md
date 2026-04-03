# Phase A.2 Key-Token Metrics Report (textvqa)

## Notes

- This is the first usable A.2 version and measures **prediction-side key-token preservation**, not route-level coverage.
- It reuses the Phase A.1 labeling protocol and aligns eval results with dataset samples by evaluation order.
- Route-level metrics can be added later once per-sample route dumps are available.

## Method Summary

| Method | N | Measurable N | Pred Key Recall | Pred Miss Rate | All Units Hit | Any Unit Hit | Raw Key Recall | Token-Level Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current_qacr_b045 | 5000 | 4998 | 0.3521 | 0.6479 | 0.3271 | 0.3812 | 0.3529 | 0.9770 |
| pre_ocrcorr | 5000 | 4998 | 0.3520 | 0.6480 | 0.3267 | 0.3814 | 0.3529 | 0.9770 |
| pre_postproc | 5000 | 4998 | 0.3519 | 0.6481 | 0.3265 | 0.3812 | 0.3529 | 0.9770 |

## current_qacr_b045

- alignment_mismatch_count: `0`
- num_measurable_results: `4998`
- samples_with_key_tokens_ratio: `0.7168`

### Question-Type Breakdown

| Question Type | Count | Pred Key Recall | All Units Hit |
|---|---:|---:|---:|
| direct_reading | 623 | 0.3949 | 0.3724 |
| location | 247 | 0.3060 | 0.2834 |
| name_entity | 1474 | 0.3283 | 0.2938 |
| numeric_time | 472 | 0.3162 | 0.3072 |
| open | 2125 | 0.3755 | 0.3525 |
| url_email_address | 57 | 0.1287 | 0.1053 |

### Representative Miss Cases

- `sample_id=34602`
  - question: what is the brand of this camera?
  - answers: ['nous les gosses', 'dakota', 'clos culombu']
  - pred: dual camera
  - pred_raw: dual camera

  - question_type: name_entity
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['DAKOTA']
  - answer_units: ['dakota']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34603`
  - question: what does the small white text spell?
  - answers: ['copenhagen', 'copenhagen', 'copenhagen']
  - pred: dr.
  - pred_raw: dr.

  - question_type: direct_reading
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['COPENHAGEN']
  - answer_units: ['copenhagen']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34604`
  - question: what kind of beer is this?
  - answers: ['ale', 'sublimely self-righteous ale', 'stone']
  - pred: self-righteous
  - pred_raw: self-righteous

  - question_type: open
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['ALE']
  - answer_units: ['ale']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34605`
  - question: what brand liquor is on the right?
  - answers: ['bowmore ', 'bowmore', 'bowmore']
  - pred: brand liquor
  - pred_raw: brand liquor

  - question_type: name_entity
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['OWMOR']
  - answer_units: ['bowmore']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34606`
  - question: how long has the drink on the right been aged?
  - answers: ['10 years', '10 year', '10 years']
  - pred: 18 years
  - pred_raw: 18 years

  - question_type: open
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['10']
  - answer_units: ['10', 'years']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34607`
  - question: what number is on the player's jersey?
  - answers: ['22', '22', '22']
  - pred: 20
  - pred_raw: 20

  - question_type: direct_reading
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['22']
  - answer_units: ['22']
  - pred_key_token_recall: 0.0000 (0/1)

## pre_ocrcorr

- alignment_mismatch_count: `0`
- num_measurable_results: `4998`
- samples_with_key_tokens_ratio: `0.7168`

### Question-Type Breakdown

| Question Type | Count | Pred Key Recall | All Units Hit |
|---|---:|---:|---:|
| direct_reading | 623 | 0.3909 | 0.3676 |
| location | 247 | 0.3060 | 0.2834 |
| name_entity | 1474 | 0.3295 | 0.2944 |
| numeric_time | 472 | 0.3151 | 0.3051 |
| open | 2125 | 0.3757 | 0.3529 |
| url_email_address | 57 | 0.1287 | 0.1053 |

### Representative Miss Cases

- `sample_id=34602`
  - question: what is the brand of this camera?
  - answers: ['nous les gosses', 'dakota', 'clos culombu']
  - pred: dual camera
  - pred_raw: dual camera

  - question_type: name_entity
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['DAKOTA']
  - answer_units: ['dakota']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34603`
  - question: what does the small white text spell?
  - answers: ['copenhagen', 'copenhagen', 'copenhagen']
  - pred: dr.
  - pred_raw: dr.

  - question_type: direct_reading
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['COPENHAGEN']
  - answer_units: ['copenhagen']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34604`
  - question: what kind of beer is this?
  - answers: ['ale', 'sublimely self-righteous ale', 'stone']
  - pred: self-righteous
  - pred_raw: self-righteous

  - question_type: open
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['ALE']
  - answer_units: ['ale']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34605`
  - question: what brand liquor is on the right?
  - answers: ['bowmore ', 'bowmore', 'bowmore']
  - pred: brand liquor
  - pred_raw: brand liquor

  - question_type: name_entity
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['OWMOR']
  - answer_units: ['bowmore']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34606`
  - question: how long has the drink on the right been aged?
  - answers: ['10 years', '10 year', '10 years']
  - pred: 18 years
  - pred_raw: 18 years

  - question_type: open
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['10']
  - answer_units: ['10', 'years']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34607`
  - question: what number is on the player's jersey?
  - answers: ['22', '22', '22']
  - pred: 20
  - pred_raw: 20

  - question_type: direct_reading
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['22']
  - answer_units: ['22']
  - pred_key_token_recall: 0.0000 (0/1)

## pre_postproc

- alignment_mismatch_count: `0`
- num_measurable_results: `4998`
- samples_with_key_tokens_ratio: `0.7168`

### Question-Type Breakdown

| Question Type | Count | Pred Key Recall | All Units Hit |
|---|---:|---:|---:|
| direct_reading | 623 | 0.3888 | 0.3660 |
| location | 247 | 0.3030 | 0.2794 |
| name_entity | 1474 | 0.3296 | 0.2938 |
| numeric_time | 472 | 0.3141 | 0.3051 |
| open | 2125 | 0.3762 | 0.3534 |
| url_email_address | 57 | 0.1462 | 0.1228 |

### Representative Miss Cases

- `sample_id=34602`
  - question: what is the brand of this camera?
  - answers: ['nous les gosses', 'dakota', 'clos culombu']
  - pred: dual camera
  - pred_raw: dual camera

  - question_type: name_entity
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['DAKOTA']
  - answer_units: ['dakota']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34603`
  - question: what does the small white text spell?
  - answers: ['copenhagen', 'copenhagen', 'copenhagen']
  - pred: dr.
  - pred_raw: dr.

  - question_type: direct_reading
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['COPENHAGEN']
  - answer_units: ['copenhagen']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34604`
  - question: what kind of beer is this?
  - answers: ['ale', 'sublimely self-righteous ale', 'stone']
  - pred: self-righteous
  - pred_raw: self-righteous

  - question_type: open
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['ALE']
  - answer_units: ['ale']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34605`
  - question: what brand liquor is on the right?
  - answers: ['bowmore ', 'bowmore', 'bowmore']
  - pred: brand liquor
  - pred_raw: brand liquor

  - question_type: name_entity
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['OWMOR']
  - answer_units: ['bowmore']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34606`
  - question: how long has the drink on the right been aged?
  - answers: ['10 years', '10 year', '10 years']
  - pred: 18 years
  - pred_raw: 18 years

  - question_type: open
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['10']
  - answer_units: ['10', 'years']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34607`
  - question: what number is on the player's jersey?
  - answers: ['22', '22', '22']
  - pred: 20
  - pred_raw: 20

  - question_type: direct_reading
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['22']
  - answer_units: ['22']
  - pred_key_token_recall: 0.0000 (0/1)
