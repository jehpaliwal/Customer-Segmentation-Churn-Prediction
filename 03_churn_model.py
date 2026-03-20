# ============================================
# PROJECT 2: Customer Segmentation & Churn
# File 3: Churn Prediction Model
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score
)

# ── Load clean data ──
df = pd.read_csv('../data/churn_clean.csv')
print(f"Loaded {len(df)} rows")

# ══════════════════════════════════════════════
# STEP 1: ENCODE CATEGORICAL COLUMNS
# ══════════════════════════════════════════════

df_model = df.copy()

le = LabelEncoder()
df_model['Geography'] = le.fit_transform(df_model['Geography'])
df_model['Gender']    = le.fit_transform(df_model['Gender'])

# ══════════════════════════════════════════════
# STEP 2: SPLIT INTO FEATURES AND TARGET
# ══════════════════════════════════════════════

X = df_model.drop('Exited', axis=1)  # everything except churn
y = df_model['Exited']               # churn column

print(f"\nFeatures shape: {X.shape}")
print(f"Target distribution:")
print(y.value_counts())
print(f"Churn rate in dataset: {y.mean()*100:.2f}%")

# ══════════════════════════════════════════════
# STEP 3: TRAIN/TEST SPLIT
# 80% for training, 20% for testing
# ══════════════════════════════════════════════

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# stratify=y means same churn ratio in both train and test

print(f"\nTraining set: {len(X_train)} rows")
print(f"Testing set:  {len(X_test)} rows")

# ══════════════════════════════════════════════
# STEP 4: SCALE FEATURES
# ══════════════════════════════════════════════

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
# IMPORTANT: fit on train, only transform on test

# ══════════════════════════════════════════════
# STEP 5: LOGISTIC REGRESSION MODEL
# ══════════════════════════════════════════════

print("\n" + "="*50)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*50)

lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_sc, y_train)
lr_pred = lr.predict(X_test_sc)
lr_proba = lr.predict_proba(X_test_sc)[:, 1]

lr_accuracy = accuracy_score(y_test, lr_pred) * 100
lr_auc = roc_auc_score(y_test, lr_proba)

print(f"Accuracy:  {lr_accuracy:.2f}%")
print(f"ROC-AUC:   {lr_auc:.3f}")
print("\nDetailed Report:")
print(classification_report(y_test, lr_pred,
      target_names=['Retained', 'Churned']))

# ══════════════════════════════════════════════
# STEP 6: RANDOM FOREST MODEL
# ══════════════════════════════════════════════

print("\n" + "="*50)
print("MODEL 2: RANDOM FOREST")
print("="*50)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1      # use all CPU cores
)
print("Training Random Forest (takes ~30 seconds)...")
rf.fit(X_train, y_train)   # RF doesn't need scaled data
rf_pred  = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred) * 100
rf_auc = roc_auc_score(y_test, rf_proba)

print(f"Accuracy:  {rf_accuracy:.2f}%")
print(f"ROC-AUC:   {rf_auc:.3f}")
print("\nDetailed Report:")
print(classification_report(y_test, rf_pred,
      target_names=['Retained', 'Churned']))

# ══════════════════════════════════════════════
# STEP 7: FEATURE IMPORTANCE
# Which factors predict churn the most?
# ══════════════════════════════════════════════

feature_names = X.columns.tolist()
feat_imp = pd.Series(
    rf.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print("\n--- TOP CHURN PREDICTORS ---")
print(feat_imp.round(4))

# ══════════════════════════════════════════════
# STEP 8: ADD CHURN PROBABILITY TO FULL DATASET
# ══════════════════════════════════════════════

# Predict churn probability for every customer
df_model_full = df.copy()
df_model_full['Geography'] = LabelEncoder().fit_transform(
    df_model_full['Geography']
)
df_model_full['Gender'] = LabelEncoder().fit_transform(
    df_model_full['Gender']
)

X_full = df_model_full.drop('Exited', axis=1)
df['Churn_Probability'] = rf.predict_proba(X_full)[:, 1]
df['Churn_Risk'] = pd.cut(
    df['Churn_Probability'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

risk_dist = df['Churn_Risk'].value_counts()
print(f"\n--- CHURN RISK DISTRIBUTION ---")
print(risk_dist)

# Save with predictions
df.to_csv('../data/churn_with_predictions.csv', index=False)
print("\nPredictions saved.")

# ══════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle('Churn Prediction Model Results', fontsize=14, fontweight='bold')

NAVY  = '#1A3C6E'
ACC   = '#2E5FAC'
RED   = '#e74c3c'
GREEN = '#2ecc71'

# Chart 1: Feature Importance
top_features = feat_imp.head(8)
colors_fi = [NAVY if i == 0 else ACC if i < 3 else '#8FBFE0'
             for i in range(len(top_features))]
axes[0, 0].barh(
    top_features.index[::-1],
    top_features.values[::-1],
    color=colors_fi[::-1], edgecolor='white'
)
axes[0, 0].set_title('Top Churn Predictors (Feature Importance)',
                     fontweight='bold')
axes[0, 0].set_xlabel('Importance Score')
axes[0, 0].spines[['top', 'right']].set_visible(False)

# Chart 2: Confusion Matrix — Random Forest
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Retained', 'Churned'],
    yticklabels=['Retained', 'Churned'],
    ax=axes[0, 1], linewidths=1
)
axes[0, 1].set_title(
    f'Confusion Matrix — Random Forest\n(Accuracy: {rf_accuracy:.1f}%)',
    fontweight='bold'
)
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# Chart 3: ROC Curve — both models
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
axes[1, 0].plot(fpr_lr, tpr_lr, color=ACC,
                label=f'Logistic Regression (AUC={lr_auc:.2f})',
                linewidth=2)
axes[1, 0].plot(fpr_rf, tpr_rf, color=NAVY,
                label=f'Random Forest (AUC={rf_auc:.2f})',
                linewidth=2)
axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random Baseline')
axes[1, 0].set_title('ROC Curve — Model Comparison', fontweight='bold')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].legend(fontsize=9)
axes[1, 0].spines[['top', 'right']].set_visible(False)

# Chart 4: Churn Risk Distribution
risk_colors = [GREEN, '#f39c12', RED]
axes[1, 1].bar(
    risk_dist.index, risk_dist.values,
    color=risk_colors, edgecolor='white'
)
for i, (label, val) in enumerate(risk_dist.items()):
    axes[1, 1].text(
        i, val + 30, f'{val:,}\n({val/len(df)*100:.1f}%)',
        ha='center', fontweight='bold', fontsize=10
    )
axes[1, 1].set_title('Customer Churn Risk Distribution',
                     fontweight='bold')
axes[1, 1].set_ylabel('Number of Customers')
axes[1, 1].spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('../outputs/model_results.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("Model results chart saved.")

# ══════════════════════════════════════════════
# PRINT SUMMARY FOR YOUR REPORT
# ══════════════════════════════════════════════

print("\n" + "="*50)
print("SUMMARY — COPY THESE INTO YOUR REPORT")
print("="*50)
print(f"Logistic Regression Accuracy: {lr_accuracy:.1f}%")
print(f"Random Forest Accuracy:       {rf_accuracy:.1f}%")
print(f"Random Forest ROC-AUC:        {rf_auc:.3f}")
print(f"Top churn predictor:          {feat_imp.index[0]}")
print(f"2nd top churn predictor:      {feat_imp.index[1]}")
print(f"High-risk customers:          {risk_dist.get('High Risk', 0):,}")
print(f"Medium-risk customers:        {risk_dist.get('Medium Risk', 0):,}")
print(f"Low-risk customers:           {risk_dist.get('Low Risk', 0):,}")
print("="*50)
print("\nCHURN MODEL COMPLETE.")