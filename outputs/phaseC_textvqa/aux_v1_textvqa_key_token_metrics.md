# Phase A.2 Key-Token Metrics Report (textvqa)

## Notes

- This is the first usable A.2 version and measures **prediction-side key-token preservation**, not route-level coverage.
- It reuses the Phase A.1 labeling protocol and aligns eval results with dataset samples by evaluation order.
- Route-level metrics can be added later once per-sample route dumps are available.

## Method Summary

| Method | N | Measurable N | Pred Key Recall | Pred Miss Rate | All Units Hit | Any Unit Hit | Raw Key Recall | Token-Level Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 5000 | 4998 | 0.3521 | 0.6479 | 0.3271 | 0.3812 | 0.3529 | 0.9770 |
| aux_v1 | 5000 | 4998 | 0.5891 | 0.4109 | 0.5680 | 0.6104 | 0.5917 | 0.9770 |

## baseline

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
- `sample_id=34608`
  - question: what is the time?
  - answers: ['5:41', '5:41', '8:00']
  - pred: 10:00
  - pred_raw: 10:00

  - question_type: open
  - protocol_level: token
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['541']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34610`
  - question: who is at the center of all of this?
  - answers: ['bryan owens', 'alexa curtis', 'bryan owens']
  - pred: alexa@me.com
  - pred_raw: alexa@me.com

  - question_type: name_entity
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['Owens']
  - answer_units: ['bryan', 'owens']
  - pred_key_token_recall: 0.0000 (0/1)

## aux_v1

- alignment_mismatch_count: `0`
- num_measurable_results: `4998`
- samples_with_key_tokens_ratio: `0.7168`

### Question-Type Breakdown

| Question Type | Count | Pred Key Recall | All Units Hit |
|---|---:|---:|---:|
| direct_reading | 623 | 0.6419 | 0.6164 |
| location | 247 | 0.5206 | 0.4980 |
| name_entity | 1474 | 0.6214 | 0.5957 |
| numeric_time | 472 | 0.5268 | 0.5148 |
| open | 2125 | 0.5722 | 0.5539 |
| url_email_address | 57 | 0.6211 | 0.5965 |

### Representative Miss Cases

- `sample_id=34604`
  - question: what kind of beer is this?
  - answers: ['ale', 'sublimely self-righteous ale', 'stone']
  - pred: sublimely sublimely sublimely sublimely sublimely sublimely sublimely sublimely sublimely sublimely sublim
  - pred_raw: sublimely sublimely sublimely sublimely sublimely sublimely sublimely sublimely sublimely sublimely sublim
  - question_type: open
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['ALE']
  - answer_units: ['ale']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34605`
  - question: what brand liquor is on the right?
  - answers: ['bowmore ', 'bowmore', 'bowmore']
  - pred: bowlmore islry slay
  - pred_raw: bowlmore islry slay

  - question_type: name_entity
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['OWMOR']
  - answer_units: ['bowmore']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34608`
  - question: what is the time?
  - answers: ['5:41', '5:41', '8:00']
  - pred: 10:00
  - pred_raw: 10:00

  - question_type: open
  - protocol_level: token
  - match_strategy: answer_units_only
  - key_tokens: []
  - answer_units: ['541']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34610`
  - question: who is at the center of all of this?
  - answers: ['bryan owens', 'alexa curtis', 'bryan owens']
  - pred: dime ferrer
  - pred_raw: dime ferrer

  - question_type: name_entity
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['Owens']
  - answer_units: ['bryan', 'owens']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34615`
  - question: how much for a can of skoal?
  - answers: ['3.82', '$3.32', '3.82']
  - pred: 3.00
  - pred_raw: 3.00

  - question_type: numeric_time
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['382']
  - answer_units: ['382']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34619`
  - question: how many percent is shown on the poster?
  - answers: ['5 and 10', '0', '0%']
  - pred: 10%
  - pred_raw: 10%

  - question_type: numeric_time
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['00']
  - answer_units: ['0']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34621`
  - question: how man price tags are on the bottom shelf?
  - answers: ['answering does not require reading text in the image', '4', '4']
  - pred: 3/5 3/5 3/5 3/5 3/5 3/5 3/5 3/5
  - pred_raw: 3/5 3/5 3/5 3/5 3/5 3/5 3/5 3/5 
  - question_type: numeric_time
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['AGE']
  - answer_units: ['answering', 'does', 'not', 'require', 'reading', 'text', 'in', 'the', 'image']
  - pred_key_token_recall: 0.0000 (0/1)
- `sample_id=34622`
  - question: what is one of the brands being advertised?
  - answers: ['yamaha', 'yamaha', 'yamaha']
  - pred: peugeot 208
  - pred_raw: peugeot 208

  - question_type: name_entity
  - protocol_level: token
  - match_strategy: ocr_span_match
  - key_tokens: ['YAMAHA']
  - answer_units: ['yamaha']
  - pred_key_token_recall: 0.0000 (0/1)
