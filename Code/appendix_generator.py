"""
Appendix Generator for SkyHack Challenge
Generates visualizations and analysis charts for the appendix
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def create_appendix_content():
    """Generate all charts and content for the appendix"""
    
    print("\n--- Generating All Content for a Winner's Appendix ---")
    
    # Load the exported data
    df = pd.read_csv('final_flight_data.csv')
    print("✅ Successfully loaded 'final_flight_data.csv'.")
    
    # Ensure Charts and TXT folders exist
    charts_dir = os.path.join(os.getcwd(), "Charts")
    txt_dir = os.path.join(os.getcwd(), "TXT")
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Chart 1: Load Factor Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['load_factor'], bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(df['load_factor'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["load_factor"].mean():.2f}')
    plt.xlabel('Load Factor')
    plt.ylabel('Frequency')
    plt.title('Distribution of Load Factor Across Flights')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'appendix_chart_1_load_factor.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Appendix Chart 1: Load Factor Distribution saved.")
    
    # Chart 2: Feature Correlation Heatmap
    feature_columns = [
        'ground_time_pressure',
        'load_factor',
        'bags_per_pax',
        'transfer_bag_pct',
        'ssr_intensity',
        'family_complexity',
        'historical_delay',
        'difficulty_score'
    ]
    
    plt.figure(figsize=(12, 10))
    corr_matrix = df[feature_columns].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'appendix_chart_2_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Appendix Chart 2: Correlation Heatmap saved.")
    
    # Chart 3: Difficulty Score by Hour of Day
    plt.figure(figsize=(12, 6))
    hourly_difficulty = df.groupby('hour')['difficulty_score'].agg(['mean', 'std'])
    plt.errorbar(hourly_difficulty.index, hourly_difficulty['mean'], 
                 yerr=hourly_difficulty['std'], marker='o', capsize=5)
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Difficulty Score')
    plt.title('Flight Difficulty Patterns Throughout the Day')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'appendix_chart_3_hourly_pattern.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Appendix Chart 3: Hourly Difficulty Pattern saved.")
    
    # Chart 4: Difficulty Class Distribution
    plt.figure(figsize=(10, 6))
    class_counts = df['difficulty_class'].value_counts()
    colors = {'Difficult': '#d62728', 'Medium': '#ff7f0e', 'Easy': '#2ca02c'}
    bars = plt.bar(class_counts.index, class_counts.values, 
                   color=[colors[x] for x in class_counts.index])
    plt.xlabel('Difficulty Class')
    plt.ylabel('Number of Flights')
    plt.title('Distribution of Flight Difficulty Classifications')
    
    total = class_counts.sum()
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}\n({height/total*100:.1f}%)',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'appendix_chart_4_difficulty_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Appendix Chart 4: Difficulty Class Distribution saved.")
    
    # Chart 5: Delay vs Difficulty Score Scatter
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(df['difficulty_score'], df['delay_minutes'], 
                         c=df['load_factor'], cmap='viridis', alpha=0.5, s=20)
    plt.colorbar(scatter, label='Load Factor')
    plt.xlabel('Difficulty Score')
    plt.ylabel('Delay (minutes)')
    plt.title('Relationship Between Difficulty Score and Flight Delays')
    
    z = np.polyfit(df['difficulty_score'].dropna(), 
                   df['delay_minutes'].dropna(), 1)
    p = np.poly1d(z)
    plt.plot(df['difficulty_score'].sort_values(), 
             p(df['difficulty_score'].sort_values()), 
             "r--", alpha=0.8, label='Trend')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'appendix_chart_5_delay_vs_difficulty.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Appendix Chart 5: Delay vs Difficulty Scatter saved.")
    
    # Chart 6: Primary Difficulty Drivers
    plt.figure(figsize=(12, 8))
    driver_counts = df[df['difficulty_class'] == 'Difficult']['primary_driver'].value_counts().head(10)
    driver_counts.plot(kind='barh', color='steelblue')
    plt.xlabel('Number of Flights')
    plt.ylabel('Primary Difficulty Driver')
    plt.title('Top 10 Primary Drivers for Difficult Flights')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'appendix_chart_6_primary_drivers.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Appendix Chart 6: Primary Drivers saved.")
    
    # Chart 7: Box Plot - Delay by Difficulty Class
    plt.figure(figsize=(10, 6))
    df_clean = df[df['delay_minutes'].notna()]
    sns.boxplot(data=df_clean, x='difficulty_class', y='delay_minutes',
                order=['Easy', 'Medium', 'Difficult'],
                palette={'Easy': '#2ca02c', 'Medium': '#ff7f0e', 'Difficult': '#d62728'})
    plt.xlabel('Difficulty Class')
    plt.ylabel('Delay (minutes)')
    plt.title('Delay Distribution by Difficulty Class')
    plt.ylim(-50, 150)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'appendix_chart_7_delay_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Appendix Chart 7: Delay Box Plot saved.")
    
    # Chart 8: Feature Importance
    plt.figure(figsize=(12, 8))
    rank_columns = [col for col in df.columns if col.endswith('_rank')]
    feature_names = [col.replace('_rank', '') for col in rank_columns]
    
    variances = [df[col].var() for col in rank_columns]
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Variance': variances
    }).sort_values('Variance', ascending=True).tail(12)
    
    feature_importance.plot(x='Feature', y='Variance', kind='barh', 
                           color='coral', legend=False)
    plt.xlabel('Rank Variance (Higher = More Discriminative)')
    plt.ylabel('Feature')
    plt.title('Feature Importance Based on Rank Variance')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'appendix_chart_8_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Appendix Chart 8: Feature Importance saved.")
    
    # Summary Statistics -> TXT folder
    summary_stats = {
        'Total Flights Analyzed': len(df),
        'Difficult Flights': (df['difficulty_class'] == 'Difficult').sum(),
        'Medium Flights': (df['difficulty_class'] == 'Medium').sum(),
        'Easy Flights': (df['difficulty_class'] == 'Easy').sum(),
        'Average Difficulty Score': df['difficulty_score'].mean(),
        'Average Delay (All Flights)': df['delay_minutes'].mean(),
        'Average Delay (Difficult)': df[df['difficulty_class'] == 'Difficult']['delay_minutes'].mean(),
        'Average Load Factor': df['load_factor'].mean(),
        'Flights with Insufficient Ground Time': (df['insufficient_ground_time'] == 1).sum(),
        'International Flights': (df['is_international'] == 1).sum(),
    }
    
    txt_path = os.path.join(txt_dir, 'appendix_summary_stats.txt')
    with open(txt_path, 'w') as f:
        f.write("=== SKYHACK CHALLENGE - SUMMARY STATISTICS ===\n\n")
        for key, value in summary_stats.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.2f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"✅ Summary statistics saved to '{txt_path}'.")
    
    print("\n✅ All appendix charts and content generated successfully!")
    print("📊 Generated 8 charts in Charts/ + 1 summary statistics file in TXT/")
    print("\nFiles created inside Charts/:")
    print("  - appendix_chart_1_load_factor.png")
    print("  - appendix_chart_2_correlation.png")
    print("  - appendix_chart_3_hourly_pattern.png")
    print("  - appendix_chart_4_difficulty_distribution.png")
    print("  - appendix_chart_5_delay_vs_difficulty.png")
    print("  - appendix_chart_6_primary_drivers.png")
    print("  - appendix_chart_7_delay_boxplot.png")
    print("  - appendix_chart_8_feature_importance.png")
    print("\nFiles created inside TXT/:")
    print("  - appendix_summary_stats.txt")

if __name__ == "__main__":
    create_appendix_content()
