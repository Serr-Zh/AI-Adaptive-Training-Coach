from pydantic import BaseModel, Field, model_validator
from typing import Optional, Literal
from enum import Enum


# Входные модели

class Goal(str, Enum):
    hypertrophy = "hypertrophy"   # гипертрофия
    strength = "strength"          # сила
    endurance = "endurance"        # выносливость
    general = "general"            # общая форма


class ExperienceLevel(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class UserProfile(BaseModel):
    goal: Goal = Field(..., description="Цель тренировок")
    experience_level: ExperienceLevel = Field(..., description="Уровень подготовки")
    equipment: list[str] = Field(..., description="Доступное оборудование, например ['штанга', 'гантели', 'турник']")
    restrictions: list[str] = Field(default_factory=list, description="Ограничения, например ['боль в колене', 'нет становой тяги']")


class ExerciseRecord(BaseModel):
    name: str = Field(..., description="Название упражнения")
    sets_planned: int = Field(..., ge=1, description="Запланировано подходов")
    sets_completed: int = Field(..., ge=0, description="Выполнено подходов")
    reps: str = Field(..., description="Повторения, например '5x3' или '8/7/6'")
    weight_kg: Optional[float] = Field(None, description="Рабочий вес в кг")
    rpe: Optional[int] = Field(None, ge=1, le=10, description="Субъективная сложность от 1 до 10")

    @model_validator(mode="after")
    def sets_completed_cannot_exceed_planned(self) -> "ExerciseRecord":
        if self.sets_completed > self.sets_planned:
            raise ValueError("sets_completed не может превышать sets_planned")
        return self


class TrainingSession(BaseModel):
    date: str = Field(..., description="Дата тренировки в формате YYYY-MM-DD")
    exercises: list[ExerciseRecord] = Field(default_factory=list)
    sleep_hours: Optional[float] = Field(None, description="Часов сна перед тренировкой")
    fatigue_level: Optional[int] = Field(None, ge=1, le=10, description="Уровень усталости от 1 до 10")
    notes: Optional[str] = Field(None, description="Произвольные заметки")


class CoachRequest(BaseModel):
    user_profile: UserProfile
    session_history: list[TrainingSession] = Field(
        default_factory=list,
        description="История тренировок от старых к новым. Пустой список — генерация стартового плана"
    )
    current_session: Optional[TrainingSession] = Field(
        None,
        description="Только что выполненная тренировка. Если передана — система адаптирует следующую"
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Температура генерации LLM. Низкая = стабильнее, высокая = вариативнее"
    )


# Выходные модели — строгая JSON-схема

class ExerciseChange(BaseModel):
    exercise_name: str
    change_type: str = Field(..., description="Например: 'снизить интенсивность', 'убрать упражнение', 'заменить на'")
    details: str = Field(..., description="Конкретное изменение: '80кг → 72кг' или 'заменить на жим гантелей'")


class NextSessionPlan(BaseModel):
    decision: str = Field(..., description="Общее решение: оставить / снизить нагрузку / повысить / deload")
    exercise_changes: list[ExerciseChange] = Field(default_factory=list, description="Конкретные изменения по упражнениям")
    reasoning: str = Field(..., description="Краткое объяснение, почему принято такое решение")


class CoachResponse(BaseModel):
    mode: Literal["initial_plan", "adaptation"] = Field(..., description="Режим работы системы")
    session_assessment: Optional[str] = Field(None, description="Оценка последней тренировки (только в режиме адаптации)")
    next_session: NextSessionPlan
    long_term_recommendation: Optional[str] = Field(None, description="Рекомендация на ближайший микроцикл")
    safety_warnings: list[str] = Field(default_factory=list, description="Предупреждения: риски, нарушения ограничений")
    refused: bool = Field(default=False, description="True если запрос выходит за безопасные границы")
    refuse_reason: Optional[str] = Field(None, description="Причина отказа если refused=True")