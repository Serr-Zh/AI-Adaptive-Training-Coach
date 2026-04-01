from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


# =========================
# Входные модели
# =========================

class Goal(str, Enum):
    hypertrophy = "hypertrophy"
    strength = "strength"
    endurance = "endurance"
    general = "general"


class ExperienceLevel(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class UserProfile(BaseModel):
    goal: Goal = Field(..., description="Цель тренировок")
    experience_level: ExperienceLevel = Field(..., description="Уровень подготовки")
    equipment: list[str] = Field(
        ...,
        description="Доступное оборудование, например ['штанга', 'гантели', 'турник']",
    )
    restrictions: list[str] = Field(
        default_factory=list,
        description="Ограничения, например ['боль в колене', 'нет становой тяги']",
    )


class ExerciseRecord(BaseModel):
    name: str = Field(..., description="Название упражнения")
    sets_planned: int = Field(..., ge=1, description="Запланировано подходов")
    sets_completed: int = Field(..., ge=0, description="Выполнено подходов")
    reps: str = Field(..., description="Повторения, например '5x3' или '8/7/6'")
    weight_kg: Optional[float] = Field(None, description="Рабочий вес в кг")
    rpe: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="Субъективная сложность от 1 до 10",
    )

    @model_validator(mode="after")
    def sets_completed_cannot_exceed_planned(self) -> "ExerciseRecord":
        if self.sets_completed > self.sets_planned:
            raise ValueError("sets_completed не может превышать sets_planned")
        return self


class TrainingSession(BaseModel):
    date: str = Field(..., description="Дата тренировки в формате YYYY-MM-DD")
    exercises: list[ExerciseRecord] = Field(default_factory=list)
    sleep_hours: Optional[float] = Field(
        None,
        description="Часов сна перед тренировкой",
    )
    fatigue_level: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="Уровень усталости от 1 до 10",
    )
    notes: Optional[str] = Field(None, description="Произвольные заметки")


class CoachRequest(BaseModel):
    user_profile: UserProfile
    session_history: list[TrainingSession] = Field(
        default_factory=list,
        description=(
            "История тренировок от старых к новым. "
            "Пустой список — генерация стартового плана"
        ),
    )
    current_session: Optional[TrainingSession] = Field(
        None,
        description=(
            "Только что выполненная тренировка. "
            "Если передана — система адаптирует следующую"
        ),
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Температура генерации LLM. "
            "Низкая = стабильнее, высокая = вариативнее"
        ),
    )


# =========================
# Выходные модели API
# =========================

class ExerciseChange(BaseModel):
    exercise_name: str
    change_type: str = Field(
        ...,
        description="Например: 'снизить интенсивность', 'убрать упражнение', 'заменить на'",
    )
    details: str = Field(
        ...,
        description="Конкретное изменение: '80кг → 72кг' или 'заменить на жим гантелей'",
    )


class NextSessionPlan(BaseModel):
    decision: str = Field(
        ...,
        description="Общее решение: оставить / снизить нагрузку / повысить / deload",
    )
    exercise_changes: list[ExerciseChange] = Field(
        default_factory=list,
        description="Конкретные изменения по упражнениям",
    )
    reasoning: str = Field(
        ...,
        description="Краткое объяснение, почему принято такое решение",
    )


class CoachResponse(BaseModel):
    mode: Literal["initial_plan", "adaptation"] = Field(
        ...,
        description="Режим работы системы",
    )
    session_assessment: Optional[str] = Field(
        None,
        description="Оценка последней тренировки (только в режиме адаптации)",
    )
    next_session: NextSessionPlan
    long_term_recommendation: Optional[str] = Field(
        None,
        description="Рекомендация на ближайший микроцикл",
    )
    safety_warnings: list[str] = Field(
        default_factory=list,
        description="Предупреждения: риски, нарушения ограничений",
    )
    refused: bool = Field(
        default=False,
        description="True если запрос выходит за безопасные границы",
    )
    refuse_reason: Optional[str] = Field(
        None,
        description="Причина отказа если refused=True",
    )


# =========================
# SGR-модели
# =========================

class ReasoningMode(str, Enum):
    initial_plan = "initial_plan"
    adaptation = "adaptation"


class PolicyType(str, Enum):
    medical_refusal = "medical_refusal"
    restriction_limited = "restriction_limited"
    overload_reduction = "overload_reduction"
    progressive_overload = "progressive_overload"
    maintain_plan = "maintain_plan"
    initial_plan_generation = "initial_plan_generation"


class FinalAction(str, Enum):
    refuse = "refuse"
    create_initial_plan = "create_initial_plan"
    increase_load = "increase_load"
    reduce_intensity = "reduce_intensity"
    reduce_volume = "reduce_volume"
    maintain = "maintain"
    modify_for_restrictions = "modify_for_restrictions"


class InputSummary(BaseModel):
    brief_goal: str = Field(
        ...,
        description="Краткое резюме цели пользователя",
    )
    experience_level: Literal["beginner", "intermediate", "advanced"] = Field(
        ...,
        description="Уровень подготовки пользователя",
    )
    equipment_summary: str = Field(
        ...,
        description="Краткое резюме доступного оборудования",
    )
    restrictions_summary: str = Field(
        ...,
        description="Краткое резюме ограничений пользователя",
    )
    has_history: bool = Field(
        ...,
        description="Есть ли история тренировок",
    )
    has_current_session: bool = Field(
        ...,
        description="Передана ли текущая тренировка",
    )


class ProgressAssessment(BaseModel):
    progress_detected: bool = Field(
        ...,
        description="Есть ли признаки хорошего восстановления и готовности к прогрессии",
    )
    supporting_facts: list[str] = Field(
        default_factory=list,
        description="Факты, указывающие на прогресс",
    )
    recommended_progression: Optional[str] = Field(
        None,
        description="Какую небольшую прогрессию можно предложить, если она допустима",
    )


class OverloadAssessment(BaseModel):
    overload_detected: bool = Field(
        ...,
        description="Есть ли признаки перегрузки",
    )
    overload_signals: list[str] = Field(
        default_factory=list,
        description="Какие именно признаки перегрузки обнаружены",
    )
    recommended_adjustment: Optional[Literal["reduce_intensity", "reduce_volume"]] = Field(
        None,
        description="Какой тип снижения нагрузки нужен при перегрузке",
    )


class MedicalRiskAssessment(BaseModel):
    medical_risk_detected: bool = Field(
        ...,
        description="Есть ли медицинский риск, требующий отказа от тренировочной рекомендации",
    )
    risk_signals: list[str] = Field(
        default_factory=list,
        description="Какие признаки указывают на медицинский риск",
    )
    refusal_required: bool = Field(
        ...,
        description="Нужно ли обязательно отказаться от выдачи тренировочной рекомендации",
    )
    refuse_reason: Optional[str] = Field(
        None,
        description="Причина отказа, если отказ обязателен",
    )


class RestrictionAssessment(BaseModel):
    restrictions_present: bool = Field(
        ...,
        description="Есть ли ограничения пользователя",
    )
    limiting_factors: list[str] = Field(
        default_factory=list,
        description="Какие ограничения влияют на решение",
    )
    restriction_impact_summary: str = Field(
        ...,
        description="Как ограничения влияют на тренировочное решение",
    )


class DecisionTrace(BaseModel):
    selected_policy: PolicyType = Field(
        ...,
        description="Какое главное правило выбрано как основа финального решения",
    )
    final_action: FinalAction = Field(
        ...,
        description="Итоговое действие системы",
    )
    policy_reasoning: str = Field(
        ...,
        description="Почему выбрано именно это правило и это действие",
    )


class SGRExerciseChange(BaseModel):
    exercise_name: str = Field(..., description="Название упражнения")
    change_type: str = Field(..., description="Тип изменения")
    details: str = Field(..., description="Детали изменения")


class FinalRecommendation(BaseModel):
    session_assessment: Optional[str] = Field(
        None,
        description="Краткая оценка текущей или последней тренировки",
    )
    decision: str = Field(
        ...,
        description="Краткое итоговое решение по следующей сессии",
    )
    exercise_changes: list[SGRExerciseChange] = Field(
        default_factory=list,
        description="Изменения по упражнениям",
    )
    reasoning: str = Field(
        ...,
        description="Краткое объяснение итогового решения",
    )
    long_term_recommendation: Optional[str] = Field(
        None,
        description="Рекомендация на ближайший микроцикл",
    )
    safety_warnings: list[str] = Field(
        default_factory=list,
        description="Предупреждения и safety-замечания",
    )
    refused: bool = Field(
        ...,
        description="Нужно ли отказаться от тренировочной рекомендации",
    )
    refuse_reason: Optional[str] = Field(
        None,
        description="Причина отказа, если refused=True",
    )


class CoachSGRResponse(BaseModel):
    mode: ReasoningMode = Field(
        ...,
        description="Определённый режим работы: initial_plan или adaptation",
    )
    input_summary: InputSummary
    progress_assessment: ProgressAssessment
    overload_assessment: OverloadAssessment
    medical_risk_assessment: MedicalRiskAssessment
    restriction_assessment: RestrictionAssessment
    decision_trace: DecisionTrace
    final_recommendation: FinalRecommendation

    @model_validator(mode="after")
    def validate_internal_consistency(self) -> "CoachSGRResponse":
        if (
            self.medical_risk_assessment.refusal_required
            and not self.final_recommendation.refused
        ):
            raise ValueError(
                "Если medical_risk_assessment.refusal_required=True, "
                "то final_recommendation.refused должен быть True"
            )

        if (
            self.final_recommendation.refused
            and not self.final_recommendation.refuse_reason
        ):
            raise ValueError(
                "Если final_recommendation.refused=True, "
                "то final_recommendation.refuse_reason должен быть заполнен"
            )

        if (
            self.final_recommendation.refused
            and self.final_recommendation.exercise_changes
        ):
            raise ValueError(
                "При refused=True список exercise_changes должен быть пустым"
            )

        if (
            self.medical_risk_assessment.medical_risk_detected
            and self.decision_trace.final_action != FinalAction.refuse
        ):
            raise ValueError(
                "Если обнаружен medical risk, final_action должен быть refuse"
            )

        return self


# =========================
# Преобразование SGR -> API response
# =========================

def sgr_to_coach_response(sgr: CoachSGRResponse) -> CoachResponse:
    return CoachResponse(
        mode=sgr.mode.value,
        session_assessment=sgr.final_recommendation.session_assessment,
        next_session=NextSessionPlan(
            decision=sgr.final_recommendation.decision,
            exercise_changes=[
                ExerciseChange(
                    exercise_name=change.exercise_name,
                    change_type=change.change_type,
                    details=change.details,
                )
                for change in sgr.final_recommendation.exercise_changes
            ],
            reasoning=sgr.final_recommendation.reasoning,
        ),
        long_term_recommendation=sgr.final_recommendation.long_term_recommendation,
        safety_warnings=sgr.final_recommendation.safety_warnings,
        refused=sgr.final_recommendation.refused,
        refuse_reason=sgr.final_recommendation.refuse_reason,
    )