# ============================================
# PROJECT 2: Customer Segmentation & Churn
# File 1: Exploratory Data Analysis (EDA)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── STEP 1: Load the data ──
print("Loading data...")
df = pd.read_csv('../data/Churn_Modelling.csv')

print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"\nColumn names:")
print(list(df.columns))

# ── STEP 2: See first 5 rows ──
print("\n--- FIRST 5 ROWS ---")
print(df.head())

# ── STEP 3: Check data types ──
print("\n--- DATA TYPES ---")
print(df.dtypes)

# ── STEP 4: Check missing values ──
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())
# Expected: 0 missing values — this is a clean dataset

# ── STEP 5: Drop columns we don't need ──
# RowNumber, CustomerId, Surname are identifiers — not useful for analysis
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
print(f"\nAfter dropping ID columns, shape: {df.shape}")

# ── STEP 6: Overall churn rate ──
total = len(df)
churned = df['Exited'].sum()
churn_rate = (churned / total) * 100
print(f"\n=== HEADLINE METRIC ===")
print(f"Total customers: {total}")
print(f"Customers who churned: {churned}")
print(f"Overall Churn Rate: {churn_rate:.2f}%")

# ── STEP 7: Churn by Geography ──
print("\n--- CHURN RATE BY GEOGRAPHY ---")
geo_churn = df.groupby('Geography')['Exited'].mean() * 100
print(geo_churn.round(2))

# ── STEP 8: Churn by Gender ──
print("\n--- CHURN RATE BY GENDER ---")
gender_churn = df.groupby('Gender')['Exited'].mean() * 100
print(gender_churn.round(2))

# ── STEP 9: Churn by Number of Products ──
print("\n--- CHURN RATE BY NUMBER OF PRODUCTS ---")
prod_churn = df.groupby('NumOfProducts')['Exited'].mean() * 100
print(prod_churn.round(2))
# KEY INSIGHT: 3-4 products = very high churn. Counterintuitive!

# ── STEP 10: Churn by Active Member status ──
print("\n--- CHURN RATE BY ACTIVE MEMBER STATUS ---")
active_churn = df.groupby('IsActiveMember')['Exited'].mean() * 100
print(active_churn.round(2))
# 0 = inactive, 1 = active

# ── STEP 11: Age stats by churn ──
print("\n--- AGE STATS BY CHURN STATUS ---")
age_stats = df.groupby('Exited')['Age'].describe()
print(age_stats)

# ── STEP 12: Balance stats by churn ──
print("\n--- BALANCE STATS BY CHURN STATUS ---")
bal_stats = df.groupby('Exited')['Balance'].describe()
print(bal_stats)

# ── SAVE CLEAN DATA ──
df.to_csv('../data/churn_clean.csv', index=False)
print("\nClean data saved to data/churn_clean.csv")

# ══════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════
print("\nGenerating EDA charts...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Customer Churn — Exploratory Data Analysis',
             fontsize=14, fontweight='bold')

NAVY  = '#1A3C6E'
RED   = '#e74c3c'
GREEN = '#2ecc71'

# Chart 1: Overall churn pie
axes[0, 0].pie(
    [total - churned, churned],
    labels=['Retained', 'Churned'],
    colors=[GREEN, RED],
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
axes[0, 0].set_title('Overall Churn Rate', fontweight='bold')

# Chart 2: Churn by Geography
geo_data = df.groupby('Geography')['Exited'].mean() * 100
bars2 = axes[0, 1].bar(
    geo_data.index, geo_data.values,
    color=[NAVY, RED, GREEN], edgecolor='white'
)
for bar, val in zip(bars2, geo_data.values):
    axes[0, 1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f'{val:.1f}%', ha='center', fontweight='bold', fontsize=10
    )
axes[0, 1].set_title('Churn Rate by Geography', fontweight='bold')
axes[0, 1].set_ylabel('Churn Rate (%)')
axes[0, 1].spines[['top', 'right']].set_visible(False)

# Chart 3: Churn by Gender
gen_data = df.groupby('Gender')['Exited'].mean() * 100
bars3 = axes[0, 2].bar(
    gen_data.index, gen_data.values,
    color=[NAVY, RED], edgecolor='white', width=0.4
)
for bar, val in zip(bars3, gen_data.values):
    axes[0, 2].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f'{val:.1f}%', ha='center', fontweight='bold', fontsize=10
    )
axes[0, 2].set_title('Churn Rate by Gender', fontweight='bold')
axes[0, 2].set_ylabel('Churn Rate (%)')
axes[0, 2].spines[['top', 'right']].set_visible(False)

# Chart 4: Age distribution — churned vs retained
churned_ages = df[df['Exited'] == 1]['Age']
retained_ages = df[df['Exited'] == 0]['Age']
axes[1, 0].hist(retained_ages, bins=30, alpha=0.6,
                color=NAVY, label='Retained', edgecolor='white')
axes[1, 0].hist(churned_ages, bins=30, alpha=0.6,
                color=RED, label='Churned', edgecolor='white')
axes[1, 0].set_title('Age Distribution: Churned vs Retained',
                     fontweight='bold')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Count')
axes[1, 0].legend()
axes[1, 0].spines[['top', 'right']].set_visible(False)

# Chart 5: Churn by number of products
prod_data = df.groupby('NumOfProducts')['Exited'].mean() * 100
colors_prod = [GREEN, GREEN, RED, RED]
bars5 = axes[1, 1].bar(
    prod_data.index.astype(str),
    prod_data.values,
    color=colors_prod[:len(prod_data)],
    edgecolor='white'
)
for bar, val in zip(bars5, prod_data.values):
    axes[1, 1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f'{val:.1f}%', ha='center', fontweight='bold', fontsize=10
    )
axes[1, 1].set_title('Churn Rate by Number of Products',
                     fontweight='bold')
axes[1, 1].set_xlabel('Number of Products')
axes[1, 1].set_ylabel('Churn Rate (%)')
axes[1, 1].spines[['top', 'right']].set_visible(False)

# Chart 6: Balance distribution — churned vs retained
axes[1, 2].hist(
    df[df['Exited'] == 0]['Balance'],
    bins=30, alpha=0.6, color=NAVY,
    label='Retained', edgecolor='white'
)
axes[1, 2].hist(
    df[df['Exited'] == 1]['Balance'],
    bins=30, alpha=0.6, color=RED,
    label='Churned', edgecolor='white'
)
axes[1, 2].set_title('Balance Distribution: Churned vs Retained',
                     fontweight='bold')
axes[1, 2].set_xlabel('Account Balance')
axes[1, 2].set_ylabel('Count')
axes[1, 2].legend()
axes[1, 2].spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('../outputs/eda_dashboard.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("EDA chart saved to outputs/eda_dashboard.png")
print("\nEDA COMPLETE.")