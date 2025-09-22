# Feasibility Scorer

A weighted quality assessment system for evaluating indicators based on the presence of baseline values, targets/goals, and time horizons.

## Features

- **Multilingual Support**: Detects Spanish and English indicator patterns
- **Quality-Based Scoring**: Weighted assessment requiring baseline and target components
- **Quantitative Detection**: Identifies numerical values, percentages, and dates
- **Comprehensive Testing**: Manually annotated dataset with precision/recall validation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from feasibility_scorer import FeasibilityScorer

scorer = FeasibilityScorer()

# Score a single indicator
result = scorer.calculate_feasibility_score(
    "Incrementar la línea base de 65% de cobertura educativa a una meta de 85% para el año 2025"
)

print(f"Score: {result.feasibility_score}")
print(f"Quality Tier: {result.quality_tier}")
print(f"Has Quantitative Baseline: {result.has_quantitative_baseline}")

# Batch scoring
indicators = [
    "línea base 50% meta 80% año 2025",
    "mejorar situación actual",
    "aumentar servicios región"
]
results = scorer.batch_score(indicators)
```

## Testing

```bash
pytest test_feasibility_scorer.py -v
```

## Quality Assessment Logic

### Minimum Requirements
- Both **baseline** and **target** components must be present for positive feasibility score
- Indicators missing either component receive score of 0.0

### Scoring Components
- **Baseline** (40% weight): línea base, baseline, valor inicial
- **Target** (40% weight): meta, objetivo, target, goal  
- **Time Horizon** (20% weight): horizonte temporal, timeline
- **Quantitative Bonus**: +20% each for quantitative baseline and target
- **Additional Elements**: +10% each for numerical patterns and dates

### Quality Tiers
- **High** (≥0.8): Complete indicators with quantitative elements
- **Medium** (≥0.5): Has baseline/target with some quantitative data
- **Low** (≥0.2): Basic baseline/target, limited quantitative elements
- **Poor** (<0.2): Very low confidence in detected components
- **Insufficient** (0.0): Missing baseline or target components

## Documentation

Access comprehensive detection rules:

```python
scorer = FeasibilityScorer()
print(scorer.get_detection_rules_documentation())
```