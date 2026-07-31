# Model Card: Asthma Risk Prediction Model

## Model details

- Framework: scikit-learn
- Architecture: preprocessing pipeline followed by `MLPClassifier`
- Task: binary classification
- Inputs: Age, Gender, Smoking Status, Medication, Peak Flow
- Output: 0 (No asthma) or 1 (Asthma)

## Intended use

This model is intended only for education and for testing AI ethics and governance tools.

## Out-of-scope use

- Medical diagnosis or treatment
- Clinical decision support
- Emergency decisions
- Insurance, employment, or eligibility decisions
- Processing identifiable patient information

## Risks and limitations

- Small or non-representative data may create poor generalization.
- Medication may leak information related to the target diagnosis.
- Bias may occur across gender, age, or smoking-status groups.
- Accuracy alone is insufficient to establish safety.
- The model has no clinical validation or regulatory approval.

## Recommended tests

- Compare false-positive and false-negative rates across groups.
- Test missing, extreme, malformed, and out-of-distribution inputs.
- Analyze feature influence and possible target leakage.
- Document dataset provenance and consent.
- Avoid real patient data in public testing.
