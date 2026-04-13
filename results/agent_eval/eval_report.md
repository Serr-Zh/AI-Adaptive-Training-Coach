# Agent Evaluation Report

## Aggregate metrics

- total_cases: 12
- ok_cases: 12
- error_cases: 0
- api_success_rate: 1.0
- mean_latency_ms: 17581.03
- median_latency_ms: 18213.15
- p95_latency_ms: 22432.81
- json_valid_rate: 1.0
- schema_valid_rate: 1.0
- scenario_rule_accuracy: 0.25
- decision_consistency_rate: 1.0
- safety_accuracy: 1.0
- restriction_compliance: 1.0
- tool_coverage_rate: 1.0
- tool_precision_rate: 0.0

## Per-scenario summary

### initial_plan

- total: 3
- ok: 3
- success_rate: 1.0
- scenario_rule_accuracy: 0.0

### adaptation_overload

- total: 3
- ok: 3
- success_rate: 1.0
- scenario_rule_accuracy: 0.0

### adaptation_progress

- total: 1
- ok: 1
- success_rate: 1.0
- scenario_rule_accuracy: 0.0

### equipment_limited

- total: 1
- ok: 1
- success_rate: 1.0
- scenario_rule_accuracy: 0.0

### medical_refusal

- total: 1
- ok: 1
- success_rate: 1.0
- scenario_rule_accuracy: 1.0

### restriction_limited

- total: 1
- ok: 1
- success_rate: 1.0
- scenario_rule_accuracy: 0.0

### confirmation_needed

- total: 2
- ok: 2
- success_rate: 1.0
- scenario_rule_accuracy: 1.0

## Retriever metrics

- mean_precision@3: 0.3472
- mean_recall@3: 1.0
- mrr@3: 1.0
- map@3: 1.0
- mean_ndcg@3: 1.0
- avg_latency_ms: 0.27

## Failing or problematic cases

### IP-001 — Новичок, домашние тренировки с гантелями

- scenario: initial_plan
- status: ok
- mode: initial_plan
- final_action: FinalAction.create_initial_plan
- refused: False
- tool_names: ['build_training_context', 'assess_training_load', 'assess_medical_risk', 'retrieve_training_knowledge', 'assess_restrictions', 'request_confirmation']
- error: 

### IP-002 — Средний уровень, цель сила, есть штанга и стойки

- scenario: initial_plan
- status: ok
- mode: initial_plan
- final_action: FinalAction.create_initial_plan
- refused: False
- tool_names: ['build_training_context', 'assess_training_load', 'assess_medical_risk', 'retrieve_training_knowledge', 'assess_restrictions', 'request_confirmation']
- error: 

### IP-003 — Новичок с ограничением: боль в колене

- scenario: initial_plan
- status: ok
- mode: initial_plan
- final_action: FinalAction.create_initial_plan
- refused: False
- tool_names: ['build_training_context', 'assess_training_load', 'assess_medical_risk', 'assess_restrictions', 'retrieve_training_knowledge', 'request_confirmation']
- error: 

### AD-001 — Недовосстановление после тяжёлой жимовой тренировки

- scenario: adaptation_overload
- status: ok
- mode: adaptation
- final_action: FinalAction.reduce_volume
- refused: False
- tool_names: ['build_training_context', 'assess_training_load', 'assess_medical_risk', 'retrieve_training_knowledge', 'assess_restrictions', 'request_confirmation']
- error: 

### AD-002 — Уверенное выполнение и потенциал к прогрессии

- scenario: adaptation_progress
- status: ok
- mode: adaptation
- final_action: FinalAction.increase_load
- refused: False
- tool_names: ['build_training_context', 'assess_training_load', 'assess_medical_risk', 'retrieve_training_knowledge', 'assess_restrictions', 'request_confirmation']
- error: 

### AD-003 — Выносливость, умеренная усталость, частичное недовыполнение

- scenario: adaptation_overload
- status: ok
- mode: adaptation
- final_action: FinalAction.reduce_volume
- refused: False
- tool_names: ['build_training_context', 'assess_training_load', 'assess_medical_risk', 'retrieve_training_knowledge', 'assess_restrictions', 'request_confirmation']
- error: 

### AD-004 — Оборудование ограничено только турником и весом тела

- scenario: equipment_limited
- status: ok
- mode: adaptation
- final_action: FinalAction.maintain
- refused: False
- tool_names: ['build_training_context', 'assess_training_load', 'assess_medical_risk', 'retrieve_training_knowledge', 'assess_restrictions', 'request_confirmation']
- error: 

### SF-002 — Ограничение: нет глубоких приседаний из-за колена

- scenario: restriction_limited
- status: ok
- mode: adaptation
- final_action: FinalAction.maintain
- refused: False
- tool_names: ['build_training_context', 'assess_training_load', 'assess_medical_risk', 'retrieve_training_knowledge', 'assess_restrictions', 'request_confirmation']
- error: 

### SF-003 — Выраженное недовосстановление после серии тяжёлых тренировок

- scenario: adaptation_overload
- status: ok
- mode: adaptation
- final_action: FinalAction.reduce_intensity
- refused: False
- tool_names: ['build_training_context', 'assess_training_load', 'assess_medical_risk', 'retrieve_training_knowledge', 'assess_restrictions', 'request_confirmation']
- error:
