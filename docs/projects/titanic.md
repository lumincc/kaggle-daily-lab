# Titanic: Complete Study from Baseline to Overfitting Diagnosis

## Overview

| Version | Core Strategy | LB Score | Key Lesson |
|---------|---------------|----------|------------|
| [v1](v1.md) | 4-feature Baseline + RF | ~0.77 | Sex is the strongest signal |
| [v2](v2.md) | Simplified strategy (8 features + single model) | 0.77990 | Small dataset: less is more |
| [v3](v3.md) | HasCabin + Title split + relaxed RF | 0.78947 | Cabin missing is a non-random signal |
| [v4](v4.md) | Candidate breakthrough (GBDT / Hard Voting) | Target 0.80+ | Single model bottleneck, need ensemble |

- **Competition**: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- **Kaggle Notebook**: [lumincode/titanic-lumincode](https://www.kaggle.com/code/lumincode/titanic-lumincode/edit)
- **Dataset**: 891 train / 418 test, binary classification (Survived)

---

## v1: Building the Baseline

[v1 detailed log](v1.md)

**Method**: Official starter tutorial, 4 raw features + RandomForest.

**Key Findings**:
- `Sex` alone can reach ~0.77
- `Pclass`, `SibSp`, `Parch` contribute limitedly
- Age/Fare/Cabin missing values not handled, significant information wasted

**Conclusion**: Baseline established, but feature utilization is extremely insufficient.

---

## v2: Overfitting Diagnosis and "Less is More"

[v2 detailed log](v2.md)

**Background**: After v1, an "advanced version" was attempted (not preserved as a standalone version) — 14 features + 5-model comparison + 4-model Soft Voting. Result: **CV 0.83 → LB 0.75**, an 8% gap.

**Diagnosis**:
- 891 rows cannot sustain 5 models + 40 parameter searches + Soft Voting
- `CabinLetter` missing rate 77%, too noisy
- `AgeBin`/`FareBin` binning loses information
- `Pclass_Sex` redundant with `Pclass+Sex`

**Simplification Strategy**:
- Features: 14 → 8
- Model: multi-model → single RandomForest
- Parameters: conservative (max_depth=5, min_samples_split=10)

**Result**: CV-LB gap reduced from 8% to 3%, LB stabilized at 0.77990.

**Key Lesson**: On small datasets, **less is more**. Fewer features + simpler model + conservative parameters → more stable.

---

## v3: Signal Mining — Cabin and Title

[v3 detailed log](v3.md)

**Problem**: v2 stalled at 0.78, unseen patterns remain in test.

**Three Optimizations**:

1. **HasCabin feature** — Cabin missing is not random; having Cabin ≈ First Class ≈ higher survival
2. **Dr separated** — Survival pattern of doctors/scholars differs from other Rare (Capt/Rev/Major)
3. **RF parameters relaxed** — max_depth 5→6 to capture more complex interactions

**Result**: LB 0.78947 (+0.00957), feature importance ranking:
`Sex > Title > FareLog > HasCabin > Pclass > Age > FamilySize > IsAlone > Embarked`

**Still unresolved**:
- Fare=0 outliers
- Single model (only RF tested)
- Pclass + HasCabin high-order interaction not explicitly constructed

---

## v4: Candidate Breakthrough Plan for 0.80+

[v4 detailed log](v4.md)

**Current gap**: ~0.01 (about 4–5 samples in test)

**Five Candidate Directions**:

| Direction | Strategy | Expected Gain |
|-----------|----------|---------------|
| 1 | Fare=0 outlier handling | +0.005 |
| 2 | Switch model — GradientBoosting | +0.005~0.01 |
| 3 | Amplify Title=Master strong signal | +0.003~0.005 |
| 4 | Pclass * HasCabin interaction feature | +0.003~0.005 |
| 5 | Multi-model Hard Voting | +0.005~0.01 |

**Recommended order**: Test GBDT alone first → then combined feature tuning → finally Hard Voting.

**Validation principle**: Change one variable at a time; keep if improved, revert if degraded.

---

## Core Methodology Summary

### 1. Overfitting Diagnosis Signals

| Signal | Threshold | Meaning |
|----------|-----------|---------|
| CV-LB gap | > 5% | Overfitting, simplify |
| Feature count / sample count | > 1:60 | Too many features, prune |
| Multi-model ensemble CV | Much higher than single model | Likely inflated; avoid Soft Voting on small data |

### 2. Small Dataset Strategy (<1000 samples)

- **Features**: Few but precise; avoid binning, high-cardinality categories, features with >50% missing
- **Model**: Single model first; ensemble with Hard Voting
- **Parameters**: Conservative (small max_depth, large min_samples_leaf)
- **Validation**: CV-LB gap as the core metric, not absolute CV score

### 3. Iterative Validation Flow

```
Baseline → Aggressive attempt → Diagnose → Simplify → Signal mining → Model swap / ensemble
   ↓              ↓               ↓           ↓            ↓              ↓
  v1         (failed)        v2 simplified   v3 optimized    v4 breakthrough
```

---

> The research methodology matters more than the score. Every failure is a valuable hypothesis test.
