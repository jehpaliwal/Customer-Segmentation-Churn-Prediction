# ============================================
# PROJECT 2: Customer Segmentation & Churn
# File 4: Executive Summary Dashboard
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Load data with segments and predictions ──
# Merge segmented + predictions
df_seg  = pd.read_csv('../data/churn_segmented.csv')
df_pred = pd.read_csv('../data/churn_with_predictions.csv')

# Add churn probability to segmented data
df = df_seg.copy()
df['Churn_Probability'] = df_pred['Churn_Probability']
df['Churn_Risk'] = df_pred['Churn_Risk']

print(f"Loaded {len(df)} rows for dashboard")

# ── COLOURS ──
NAVY  = '#1A3C6E'
ACC   = '#2E5FAC'
RED   = '#e74c3c'
GREEN = '#2ecc71'
AMBER = '#f39c12'
BLUES = ['#1A3C6E', '#2E5FAC', '#4A7FC1', '#8FBFE0']

# ══════════════════════════════════════════════
# BUILD 6-PANEL EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════

fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    'Customer Segmentation & Churn Analysis — Executive Dashboard',
    fontsize=16, fontweight='bold', y=0.98
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

seg_labels_short = ['Seg 0', 'Seg 1', 'Seg 2', 'Seg 3']
seg_colors = [NAVY, GREEN, RED, AMBER]

# ── Panel 1: Churn Rate by Segment ──
ax1 = fig.add_subplot(gs[0, 0])
seg_churn = df.groupby('Segment')['Exited'].mean() * 100
bars1 = ax1.bar(
    seg_labels_short, seg_churn.values,
    color=seg_colors, edgecolor='white', width=0.55
)
for bar, val in zip(bars1, seg_churn.values):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.4,
        f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11
    )
ax1.set_title('Churn Rate by Segment', fontweight='bold', fontsize=11)
ax1.set_ylabel('Churn Rate (%)')
ax1.spines[['top', 'right']].set_visible(False)

# ── Panel 2: Segment Size (Pie) ──
ax2 = fig.add_subplot(gs[0, 1])
seg_counts = df['Segment'].value_counts().sort_index()
ax2.pie(
    seg_counts.values,
    labels=seg_labels_short,
    colors=seg_colors,
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
ax2.set_title('Customer Segment Distribution', fontweight='bold', fontsize=11)

# ── Panel 3: Average Balance by Segment ──
ax3 = fig.add_subplot(gs[0, 2])
seg_balance = df.groupby('Segment')['Balance'].mean()
bars3 = ax3.bar(
    seg_labels_short, seg_balance.values,
    color=seg_colors, edgecolor='white', width=0.55
)
for bar, val in zip(bars3, seg_balance.values):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 300,
        f'£{val:,.0f}', ha='center', fontsize=9, fontweight='bold'
    )
ax3.set_title('Avg Account Balance by Segment',
              fontweight='bold', fontsize=11)
ax3.set_ylabel('Average Balance (£)')
ax3.spines[['top', 'right']].set_visible(False)

# ── Panel 4: Churn Risk Distribution ──
ax4 = fig.add_subplot(gs[1, 0])
risk_dist = df['Churn_Risk'].value_counts()
risk_order = ['Low Risk', 'Medium Risk', 'High Risk']
risk_vals = [risk_dist.get(r, 0) for r in risk_order]
risk_colors = [GREEN, AMBER, RED]
bars4 = ax4.bar(risk_order, risk_vals,
                color=risk_colors, edgecolor='white', width=0.5)
for bar, val in zip(bars4, risk_vals):
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 20,
        f'{val:,}\n({val/len(df)*100:.1f}%)',
        ha='center', fontweight='bold', fontsize=10
    )
ax4.set_title('Churn Risk Distribution', fontweight='bold', fontsize=11)
ax4.set_ylabel('Number of Customers')
ax4.spines[['top', 'right']].set_visible(False)

# ── Panel 5: Churn Rate by Geography ──
ax5 = fig.add_subplot(gs[1, 1])
geo_churn = df.groupby('Geography')['Exited'].mean() * 100
geo_colors = [NAVY, RED, GREEN]
bars5 = ax5.bar(
    geo_churn.index, geo_churn.values,
    color=geo_colors[:len(geo_churn)], edgecolor='white', width=0.5
)
for bar, val in zip(bars5, geo_churn.values):
    ax5.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11
    )
ax5.set_title('Churn Rate by Geography', fontweight='bold', fontsize=11)
ax5.set_ylabel('Churn Rate (%)')
ax5.spines[['top', 'right']].set_visible(False)

# ── Panel 6: Avg Churn Probability by Segment ──
ax6 = fig.add_subplot(gs[1, 2])
seg_prob = df.groupby('Segment')['Churn_Probability'].mean() * 100
bars6 = ax6.bar(
    seg_labels_short, seg_prob.values,
    color=seg_colors, edgecolor='white', width=0.55
)
for bar, val in zip(bars6, seg_prob.values):
    ax6.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11
    )
ax6.set_title('Avg Churn Probability by Segment',
              fontweight='bold', fontsize=11)
ax6.set_ylabel('Avg Churn Probability (%)')
ax6.spines[['top', 'right']].set_visible(False)

plt.savefig('../outputs/executive_dashboard.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("Executive dashboard saved to outputs/executive_dashboard.png")
print("\nDASHBOARD COMPLETE.")