from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

TARGET_EXAMPLES = 120
RAW_INVALID_EXAMPLES = 6
MAX_ALLOWED_DEVIATION_PP = 5.0
MAX_ALLOWED_SAFETY_DEVIATION_PP = 3.0
GENERATION_VERSION = "1.1.0"

GOALS = ["hypertrophy", "strength", "endurance", "general"]
EXPERIENCE_LEVELS = ["beginner", "intermediate", "advanced"]
POLICIES = [
    "initial_plan_generation",
    "progressive_overload",
    "overload_reduction",
    "restriction_limited",
    "maintain_plan",
    "medical_refusal",
]
FINAL_ACTIONS = [
    "refuse",
    "create_initial_plan",
    "increase_load",
    "reduce_intensity",
    "reduce_volume",
    "maintain",
    "modify_for_restrictions",
]

TARGET_DISTRIBUTIONS: Dict[str, Dict[str, float]] = {
    "mode": {
        "initial_plan": 0.35,
        "adaptation": 0.65,
    },
    "feature": {
        "initial_plan_generation": 0.35,
        "progressive_overload": 0.15,
        "overload_reduction": 0.20,
        "restriction_handling": 0.10,
        "maintain_plan": 0.10,
        "medical_safety_refusal": 0.10,
    },
    "goal": {
        "hypertrophy": 0.30,
        "strength": 0.30,
        "endurance": 0.20,
        "general": 0.20,
    },
    "experience_level": {
        "beginner": 0.35,
        "intermediate": 0.45,
        "advanced": 0.20,
    },
    "equipment_type": {
        "bodyweight": 0.20,
        "home_minimal": 0.25,
        "gym": 0.40,
        "mixed": 0.15,
    },
    "restriction_type": {
        "none": 0.40,
        "knee": 0.15,
        "lower_back": 0.15,
        "shoulder_neck": 0.15,
        "schedule_recovery": 0.15,
    },
    "session_signal": {
        "no_history": 0.35,
        "progress": 0.15,
        "overload": 0.20,
        "neutral": 0.20,
        "incomplete": 0.10,
    },
    "safety_case": {
        "safe": 0.75,
        "ambiguous_discomfort": 0.15,
        "medical_refusal": 0.10,
    },
    "expected_policy": {
        "initial_plan_generation": 0.35,
        "progressive_overload": 0.15,
        "overload_reduction": 0.20,
        "restriction_limited": 0.10,
        "maintain_plan": 0.10,
        "medical_refusal": 0.10,
    },
    "language": {
        "ru": 0.90,
        "en": 0.10,
    },
}

EQUIPMENT_BY_TYPE: Dict[str, List[List[str]]] = {
    "bodyweight": [
        ["вес тела", "турник"],
        ["bodyweight", "pull-up bar"],
        ["вес тела", "эспандер"],
    ],
    "home_minimal": [
        ["гантели", "скамья", "эспандер"],
        ["dumbbells", "bench", "bands"],
        ["гири", "коврик", "турник"],
    ],
    "gym": [
        ["штанга", "блины", "стойки", "скамья"],
        ["barbell", "plates", "rack", "bench"],
        ["тренажёры", "гантели", "штанга", "кроссовер"],
    ],
    "mixed": [
        ["гантели", "штанга", "турник", "эспандер"],
        ["dumbbells", "barbell", "pull-up bar", "bands"],
        ["гири", "гантели", "велотренажёр", "турник"],
    ],
}

RESTRICTIONS_BY_TYPE: Dict[str, List[List[str]]] = {
    "none": [[]],
    "knee": [
        ["дискомфорт в колене", "без глубоких приседаний"],
        ["knee discomfort", "no deep squats"],
        ["ограничение по колену", "избегать прыжков"],
    ],
    "lower_back": [
        ["поясница быстро устаёт", "без тяжёлой становой тяги"],
        ["lower back sensitivity", "avoid heavy deadlifts"],
        ["ограничение по пояснице", "минимум осевой нагрузки"],
    ],
    "shoulder_neck": [
        ["дискомфорт в плече", "без жима над головой"],
        ["neck stiffness", "avoid heavy overhead press"],
        ["ограничение по плечу", "избегать болезненной амплитуды"],
    ],
    "schedule_recovery": [
        ["только 45 минут на тренировку", "нужен короткий план"],
        ["частые командировки", "нет стабильного графика"],
        ["низкое восстановление после работы", "нужен умеренный объём"],
    ],
}

EXERCISES_BY_GOAL: Dict[str, Dict[str, List[Tuple[str, Optional[float]]]]] = {
    "strength": {
        "gym": [("Присед", 100.0), ("Жим лёжа", 75.0), ("Становая тяга", 130.0)],
        "mixed": [("Жим лёжа", 70.0), ("Подтягивания", None), ("Румынская тяга", 80.0)],
        "home_minimal": [("Жим гантелей", 24.0), ("Тяга гантели", 30.0), ("Гоблет-присед", 28.0)],
        "bodyweight": [("Подтягивания", None), ("Отжимания", None), ("Болгарский сплит-присед", None)],
    },
    "hypertrophy": {
        "gym": [("Жим гантелей", 28.0), ("Тяга верхнего блока", 55.0), ("Жим ногами", 120.0)],
        "mixed": [("Жим гантелей", 24.0), ("Подтягивания", None), ("Разведения гантелей", 12.0)],
        "home_minimal": [("Жим гантелей", 20.0), ("Тяга гантели", 26.0), ("Выпады назад", 12.0)],
        "bodyweight": [("Отжимания", None), ("Подтягивания", None), ("Ягодичный мост", None)],
    },
    "endurance": {
        "gym": [("Гребной тренажёр", None), ("Жим ногами", 80.0), ("Фермерская прогулка", 24.0)],
        "mixed": [("Махи гирей", 20.0), ("Подтягивания", None), ("Велотренажёр", None)],
        "home_minimal": [("Махи гирей", 16.0), ("Берпи без прыжка", None), ("Планка", None)],
        "bodyweight": [("Приседания с весом тела", None), ("Отжимания", None), ("Планка", None)],
    },
    "general": {
        "gym": [("Тяга горизонтального блока", 45.0), ("Жим гантелей", 20.0), ("Гиперэкстензия", None)],
        "mixed": [("Подтягивания", None), ("Гоблет-присед", 20.0), ("Жим гантелей", 18.0)],
        "home_minimal": [("Гоблет-присед", 16.0), ("Тяга гантели", 20.0), ("Отжимания", None)],
        "bodyweight": [("Отжимания", None), ("Подтягивания", None), ("Приседания с весом тела", None)],
    },
}

POLICY_TO_FEATURE = {
    "initial_plan_generation": "initial_plan_generation",
    "progressive_overload": "progressive_overload",
    "overload_reduction": "overload_reduction",
    "restriction_limited": "restriction_handling",
    "maintain_plan": "maintain_plan",
    "medical_refusal": "medical_safety_refusal",
}

POLICY_BLOCKS = [
    ("initial_plan_generation", 42),
    ("progressive_overload", 18),
    ("overload_reduction", 24),
    ("restriction_limited", 12),
    ("maintain_plan", 12),
    ("medical_refusal", 12),
]

NON_NONE_RESTRICTION_POOL = (
    ["knee"] * 18
    + ["lower_back"] * 18
    + ["shoulder_neck"] * 18
    + ["schedule_recovery"] * 18
)
