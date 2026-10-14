# GeoResponse v0.1 — MWD Rock-Type Classification Benchmark

## 📖 Overview

**GeoResponse** is a Measure-While-Drilling (MWD) telemetry ingestion and rock-type classification benchmark module within BlastOpt Botswana. It uses drilling telemetry (penetration rate, rotation torque, weight on bit, percussion pressure, flushing pressure) to classify rock mass strata into categorical strength units.

---

## ⚠️ Important Licensing & Deployment Notice

> **RESEARCH PROTOTYPE NOTICE:** GeoResponse is trained on the publicly documented Norwegian tunnel drilling research dataset for pipeline validation. It is **NOT** calibrated for Botswana open-pit geology (e.g. Jwaneng, Orapa, Karowe) and must **NOT** be used for operational blast design decisions or autonomous charging adjustments.

---

## 🤖 Training GeoResponse in the ML Model Manager

GeoResponse can be trained directly through the BlasterOPT **ML Model Manager**:

1. Navigate to **🤖 ML Model Manager** in the sidebar.
2. Select **Research Prototypes** under the Model Family radio buttons.
3. Choose **GeoResponse (MWD Rock Classification)**.
4. Upload an MWD CSV dataset containing drilling telemetry and rock class labels.
5. Click **Train Model** to fit the classifier on MWD Mechanical Specific Energy features.

---

## 📊 Classification Metrics vs. Regression Models

Unlike fragmentation or PPV models that predict continuous numeric targets evaluated via $R^2$, RMSE, and MAE, GeoResponse is a **classifier**. Attempting to compute $R^2$ on class labels is a category error.

GeoResponse evaluation metrics displayed in the ML Model Manager include:

- **Macro-F1 Score:** Unweighted mean F1 score across all rock classes, treating imbalanced strata equally.
- **Balanced Accuracy:** Average of recall obtained on each class, correcting for majority-class dominance.
- **Per-Class Precision, Recall, and F1 Breakdown:** Detailed performance per rock class.
- **Confusion Matrix:** True vs. predicted classification counts.
- **Majority-Class Baseline Comparison:** Performance gain ($\Delta F_1$) over a dummy classifier that always predicts the most frequent rock class.

---

## 🔬 Model Comparison & Registry Separation

In the **🔬 Model Comparison** screen:

- **Production Regression Models** (GA-ANN, ANN-RF Ensemble, PSO-ANN, PINN, baselines) are compared on $R^2$, RMSE, and MAE.
- **Research Prototypes & Classifiers** (GeoResponse) are listed in a separate **Research Prototypes** section and are **NEVER** ranked against production regression models.

---

## 🔄 Retraining Workflow for Botswana Open-Pit MWD Telemetry

To calibrate GeoResponse for site-specific Botswana open-pit geology:

1. **Ingest Real Mine MWD Data:** Ingest Sandvik (My Sandvik) or Epiroc (Certiq) ISO 15143-3 MWD CSV logs containing `penetration_rate_m_hr`, `torque_nm`, `weight_on_bit_kg`, `percussion_pressure_bar`, `flushing_pressure_bar`, and verified core log labels (`rock_class`).
2. **Group-Based Validation:** Use `GroupKFold` cross-validation grouped by `tunnel_id`, `blast_id`, or `bench_id` to prevent observation leakage across spatial splits.
3. **Review Flag Threshold:** Predictions below **70% confidence** trigger review warnings (`⚠️ Low Confidence (< 70%). Review measured core log before adjusting design.`).
