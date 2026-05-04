import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

OUT_DIR = Path("data/example")
OUT_DIR.mkdir(parents=True, exist_ok=True)

n_participants = 20
timepoints = {
    "T1": 21,    # ~3 weeks
    "T2": 42,    # ~6 weeks
    "T3": 90,    # ~3 months
    "T4": 180,   # ~6 months
}

rows = []

for pid in range(1, n_participants + 1):
    age = np.random.randint(35, 82)
    sex = np.random.choice(["Female", "Male"])
    stroke_type = np.random.choice(["infarct", "haemorrhage"], p=[0.55, 0.45])
    dominant_affected = np.random.choice([0, 1], p=[0.75, 0.25])
    mep_status = np.random.choice(["positive", "negative", "missing"], p=[0.65, 0.25, 0.10])
    inpatient_duration = np.random.randint(10, 90)

    baseline_fma = np.clip(np.random.normal(35, 22), 0, 66)
    baseline_arat = np.clip(baseline_fma / 66 * 57 + np.random.normal(0, 7), 0, 57)

    recovery_potential = np.random.uniform(0.2, 0.9)

    for tp, days in timepoints.items():
        # simple nonlinear improvement pattern
        recovery_fraction = 1 - np.exp(-days / 60)

        fma_total = baseline_fma + recovery_potential * recovery_fraction * (66 - baseline_fma)
        fma_total += np.random.normal(0, 3)
        fma_total = np.clip(fma_total, 0, 66)

        arat = baseline_arat + recovery_potential * recovery_fraction * (57 - baseline_arat)
        arat += np.random.normal(0, 4)
        arat = np.clip(arat, 0, 57)

        grip_strength = np.clip(fma_total / 66 * 25 + np.random.normal(0, 3), 0, 40)
        safe_score = np.clip(fma_total / 66 * 10 + np.random.normal(0, 1), 0, 10)
        mas_finger = np.random.choice([0, 1, 1.5, 2, 3], p=[0.25, 0.35, 0.15, 0.20, 0.05])
        moca = np.clip(np.random.normal(25, 4), 10, 30)

        # wearable metrics linked imperfectly to capacity
        faur = 0.25 + 0.7 * (arat / 57) + np.random.normal(0, 0.15)
        faur = np.clip(faur, 0.05, 1.3)

        faud = 5 + 25 * (arat / 57) + np.random.normal(0, 5)
        faud = np.clip(faud, 0, 60)

        faui = 4 + 18 * (arat / 57) + np.random.normal(0, 4)
        faui = np.clip(faui, 0, 50)

        wear_time_hours = np.clip(np.random.normal(48, 4), 24, 72)
        awake_time_hours = np.clip(np.random.normal(28, 3), 16, 36)

        rows.append({
            "participant_id": pid,
            "timepoint": tp,
            "days_post_stroke": days + np.random.randint(-7, 8),

            "age": age,
            "sex": sex,
            "stroke_type": stroke_type,
            "dominant_arm_affected": dominant_affected,
            "mep_status": mep_status,
            "inpatient_duration_days": inpatient_duration,

            "fma_total": round(fma_total, 1),
            "arat_total": round(arat, 1),
            "safe_score": round(safe_score, 1),
            "grip_strength_kg": round(grip_strength, 1),
            "mas_finger": mas_finger,
            "moca": round(moca, 1),

            "faur": round(faur, 3),
            "faud_percent": round(faud, 2),
            "faui_counts_per_second": round(faui, 2),
            "wear_time_hours": round(wear_time_hours, 1),
            "awake_time_hours": round(awake_time_hours, 1),
        })

df = pd.DataFrame(rows)

# Add some realistic missingness
for col in ["faur", "faud_percent", "faui_counts_per_second"]:
    missing_idx = df.sample(frac=0.08, random_state=42).index
    df.loc[missing_idx, col] = np.nan

df.to_csv(OUT_DIR / "mock_longitudinal_dataset.csv", index=False)

print(f"Saved mock dataset to: {OUT_DIR / 'mock_longitudinal_dataset.csv'}")
print(df.head())