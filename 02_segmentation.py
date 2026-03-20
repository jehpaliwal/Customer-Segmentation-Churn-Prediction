# ============================================
# PROJECT 2: Customer Segmentation & Churn
# File 2: K-Means Customer Segmentation
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

# ── Load clean data ──
df = pd.read_csv('../data/churn_clean.csv')
print(f"Loaded {len(df)} rows")

# ══════════════════════════════════════════════
# STEP 1: PREPARE DATA FOR CLUSTERING
# K-Means only works with numbers — encode text columns
# ══════════════════════════════════════════════

df_cluster = df.copy()

# Encode Geography (France=0, Germany=1, Spain=2)
le_geo = LabelEncoder()
df_cluster['Geography_enc'] = le_geo.fit_transform(df_cluster['Geography'])

# Encode Gender (Female=0, Male=1)
le_gen = LabelEncoder()
df_cluster['Gender_enc'] = le_gen.fit_transform(df_cluster['Gender'])

# Select features for clustering
# We use customer profile features — NOT the Exited column
features = [
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'IsActiveMember',
    'EstimatedSalary',
    'Geography_enc',
    'Gender_enc'
]

X = df_cluster[features]

print(f"\nFeatures used for clustering: {features}")
print(f"Shape: {X.shape}")

# ══════════════════════════════════════════════
# STEP 2: SCALE THE DATA
# K-Means is distance-based — all features must be
# on same scale, otherwise Balance (0-250,000) will
# dominate Age (18-90) completely
# ══════════════════════════════════════════════

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nData scaled successfully.")

# ══════════════════════════════════════════════
# STEP 3: FIND OPTIMAL NUMBER OF CLUSTERS
# Using the Elbow Method
# ══════════════════════════════════════════════

print("\nRunning Elbow Method (this takes 1-2 minutes)...")
inertias = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42,
                    n_init=10, max_iter=300)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    print(f"  K={k} done, inertia={kmeans.inertia_:.0f}")

# Plot elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, 'bo-', linewidth=2,
         markersize=8, color='#1A3C6E')
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
plt.title('Elbow Method — Finding Optimal Number of Clusters',
          fontsize=13, fontweight='bold')
plt.xticks(k_range)
plt.grid(True, alpha=0.3)
# Add annotation for chosen K
plt.axvline(x=4, color='red', linestyle='--', alpha=0.7,
            label='Chosen K=4')
plt.legend()
plt.tight_layout()
plt.savefig('../outputs/elbow_plot.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("Elbow plot saved.")

# ══════════════════════════════════════════════
# STEP 4: APPLY K-MEANS WITH K=4
# The elbow typically bends around 3-4 for this dataset
# ══════════════════════════════════════════════

print("\nApplying K-Means with K=4...")
kmeans_final = KMeans(n_clusters=4, random_state=42,
                      n_init=10, max_iter=300)
df['Segment'] = kmeans_final.fit_predict(X_scaled)
print("Clustering complete.")
print(f"\nCustomers per segment:")
print(df['Segment'].value_counts().sort_index())

# ══════════════════════════════════════════════
# STEP 5: PROFILE EACH SEGMENT
# ══════════════════════════════════════════════

segment_profile = df.groupby('Segment').agg(
    count         = ('Exited', 'count'),
    churn_rate    = ('Exited', 'mean'),
    avg_age       = ('Age', 'mean'),
    avg_balance   = ('Balance', 'mean'),
    avg_credit    = ('CreditScore', 'mean'),
    avg_tenure    = ('Tenure', 'mean'),
    avg_products  = ('NumOfProducts', 'mean'),
    pct_active    = ('IsActiveMember', 'mean'),
    avg_salary    = ('EstimatedSalary', 'mean')
).reset_index()

segment_profile['churn_rate'] = (
    segment_profile['churn_rate'] * 100
).round(1)
segment_profile['pct_of_total'] = (
    segment_profile['count'] / len(df) * 100
).round(1)
segment_profile['pct_active'] = (
    segment_profile['pct_active'] * 100
).round(1)

for col in ['avg_age', 'avg_balance', 'avg_credit',
            'avg_tenure', 'avg_products', 'avg_salary']:
    segment_profile[col] = segment_profile[col].round(1)

print("\n=== SEGMENT PROFILES ===")
print(segment_profile.to_string(index=False))

# ══════════════════════════════════════════════
# STEP 6: NAME YOUR SEGMENTS
# Look at the output and name them based on what you see
# Common patterns in this dataset:
# - High balance + high churn = At-Risk Wealthy
# - Young + low balance = Growth Potential
# - Active + low churn = Premium Loyal
# - Inactive + mid balance = Disengaged Mid-Tier
#
# YOUR SEGMENT NAMES WILL DEPEND ON YOUR OUTPUT
# Look at churn_rate and avg_balance to decide names
# ══════════════════════════════════════════════

# AFTER YOU SEE THE OUTPUT, fill in the names below:
segment_names = {
    0: 'Low-Balance Actives',
    1: 'Premium Engaged',
    2: 'At-Risk Mid-Tier',
    3: 'High-Value Disengaged'
}
# Example names (you may need to swap):
# 'Premium Loyal'       — low churn, high balance, high active%
# 'Growth Potential'    — young, low balance, medium churn
# 'At-Risk Seniors'     — older, higher churn, high balance
# 'Disengaged Mid-Tier' — inactive, medium churn, medium balance

df['Segment_Name'] = df['Segment'].map(segment_names)

# ── Save segmented data ──
df.to_csv('../data/churn_segmented.csv', index=False)
segment_profile.to_csv('../outputs/segment_profiles.csv', index=False)
print("\nSegmented data saved.")

# ══════════════════════════════════════════════
# VISUALISATION: Segment comparison
# ══════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Customer Segment Profiles', fontsize=14, fontweight='bold')

colors = ['#1A3C6E', '#2ecc71', '#e74c3c', '#f39c12']
seg_labels = [f'Seg {i}' for i in segment_profile['Segment']]

# Chart 1: Churn rate by segment
bars1 = axes[0].bar(
    seg_labels, segment_profile['churn_rate'],
    color=colors, edgecolor='white'
)
for bar, val in zip(bars1, segment_profile['churn_rate']):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f'{val:.1f}%', ha='center', fontweight='bold'
    )
axes[0].set_title('Churn Rate by Segment', fontweight='bold')
axes[0].set_ylabel('Churn Rate (%)')
axes[0].spines[['top', 'right']].set_visible(False)

# Chart 2: % of customers in each segment
bars2 = axes[1].bar(
    seg_labels, segment_profile['pct_of_total'],
    color=colors, edgecolor='white'
)
for bar, val in zip(bars2, segment_profile['pct_of_total']):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f'{val:.1f}%', ha='center', fontweight='bold'
    )
axes[1].set_title('Share of Customer Base', fontweight='bold')
axes[1].set_ylabel('% of Total Customers')
axes[1].spines[['top', 'right']].set_visible(False)

# Chart 3: Average balance by segment
bars3 = axes[2].bar(
    seg_labels, segment_profile['avg_balance'],
    color=colors, edgecolor='white'
)
for bar, val in zip(bars3, segment_profile['avg_balance']):
    axes[2].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 300,
        f'£{val:,.0f}', ha='center', fontsize=9, fontweight='bold'
    )
axes[2].set_title('Average Balance by Segment', fontweight='bold')
axes[2].set_ylabel('Average Balance (£)')
axes[2].spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('../outputs/segment_profiles_chart.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("Segment chart saved.")
print("\nSEGMENTATION COMPLETE.")