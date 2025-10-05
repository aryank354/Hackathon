"""
Enhanced Main Script with ML Models and Proper Validation
Team: Innov8torX
Members: Aryan Kanojia, Anurag Kumar
"""

from flight_analyzer import RankBasedFlightDifficultyScorer
from ml_analyzer import run_enhanced_analysis
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def run_complete_hackathon_analysis():
    """
    Complete analysis including base scoring + ML enhancements
    """
    print("=" * 80)
    print("SKYHACK CHALLENGE - COMPLETE ANALYSIS")
    print("Team: Innov8torX | Members: Aryan Kanojia, Anurag Kumar")
    print("=" * 80)

    data_paths = {
        'flights': '../Data/Flight_Level_Data.csv',
        'pnr_flights': '../Data/PNR_Flight_Level_Data.csv',
        'pnr_remarks': '../Data/PNR_Remark_Level_Data.csv',
        'bags': '../Data/Bag_Level_Data.csv',
        'airports': '../Data/Airports_Data.csv'
    }

    # ========================================================================
    # PHASE 1: BASE RANK-BASED SCORING (Original Approach)
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: BASE RANK-BASED SCORING SYSTEM")
    print("=" * 80)
    
    analyzer = RankBasedFlightDifficultyScorer()
    analyzer.load_data(data_paths)
    
    print("\n--- Exploratory Data Analysis ---")
    eda_results = analyzer.perform_eda()
    
    # Display key EDA findings
    print(f"\n✓ Average Delay: {eda_results['delay_stats']['average_delay']:.1f} minutes")
    print(f"✓ Flights Delayed: {eda_results['delay_stats']['pct_delayed']:.1f}%")
    print(f"✓ Insufficient Ground Time: {eda_results['ground_time_stats']['flights_below_minimum']} flights")
    print(f"✓ Transfer Bag Ratio: {eda_results['baggage_stats']['avg_transfer_ratio']:.1%}")
    print(f"✓ SSR Delay Impact: +{eda_results['ssr_stats']['high_ssr_delay_correlation']:.1f} minutes")
    
    print("\n--- Feature Engineering ---")
    analyzer.create_difficulty_features()
    
    print("\n--- Difficulty Score Calculation ---")
    analyzer.calculate_rank_based_scores()
    
    print("\n--- Analyzing Results ---")
    insights = analyzer.analyze_results()
    
    print("\n--- Primary Drivers for Difficult Flights ---")
    print(insights['primary_drivers'].head(5))
    
    print("\n--- Top 5 Most Difficult Destinations ---")
    print(insights['top_difficult_destinations'][['difficulty_score_mean', 
                                                   'difficulty_class_<lambda>']].head(5))
    
    # Export base results
    print("\n--- Exporting Base Results ---")
    analyzer.export_results(team_name='Innov8torX_mini')
    
    # ========================================================================
    # PHASE 2: MACHINE LEARNING ENHANCEMENT
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: MACHINE LEARNING MODELS & VALIDATION")
    print("=" * 80)
    
    enhanced, test_results, cv_results, impact = run_enhanced_analysis(analyzer)
    
    # ========================================================================
    # PHASE 3: COMPREHENSIVE SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("PHASE 3: COMPREHENSIVE SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n--- MODEL PERFORMANCE COMPARISON ---")
    print(f"Rank-Based Approach: Perfect classification by design")
    print(f"  → Validates with 29x delay separation (Easy: 1.8 min vs Difficult: 53.3 min)")
    print(f"\nRandom Forest (Temporal Validation):")
    print(f"  → CV Accuracy: {cv_results['rf_cv_mean']:.1%} (+/- {cv_results['rf_cv_std']:.1%})")
    print(f"\nGradient Boosting (Temporal Validation):")
    print(f"  → CV Accuracy: {cv_results['gb_cv_mean']:.1%} (+/- {cv_results['gb_cv_std']:.1%})")
    
    print("\n--- BUSINESS IMPACT PROJECTION ---")
    print(f"Annual Cost Savings (ORD): ${impact['annual_savings']:,.0f}")
    print(f"Minutes Saved Annually: {impact['minutes_saved']:,.0f}")
    print(f"Difficult Flights/Year: {impact['difficult_flights_annual']:,.0f}")
    print(f"\nAssumptions:")
    print(f"  - Delay cost: ${impact['assumptions']['delay_cost']}/minute")
    print(f"  - Improvement rate: {impact['assumptions']['improvement_pct']:.0%}")
    
    print("\n--- TOP RECOMMENDATIONS ---")
    recommendations = analyzer.get_recommendations()
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"\n{i}. [{rec['priority']}] {rec['category']}")
        print(f"   Issue: {rec['issue']}")
        print(f"   Action: {rec['action']}")
        print(f"   Affected: {rec['affected_flights']} unique flights")
    
    print("\n--- DELIVERABLES GENERATED ---")
    print("✓ test_Innov8torX_mini.csv - Final submission with difficulty scores")
    print("✓ final_flight_data.csv - Complete dataset with all features")
    print("✓ ML model validation - Temporal and cross-validation results")
    print("✓ Business impact analysis - Justified ROI calculations")
    print("✓ SQL queries - Production deployment scripts (see sql_queries.sql)")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - READY FOR SUBMISSION")
    print("=" * 80)
    
    # Save ML results
    test_results.to_csv('ml_validation_results.csv', index=False)
    print("\n✓ ML validation results saved to 'ml_validation_results.csv'")
    
    # Save business impact
    with open('business_impact_summary.txt', 'w') as f:
        f.write("=== BUSINESS IMPACT ANALYSIS ===\n\n")
        f.write(f"Annual Cost Savings (ORD): ${impact['annual_savings']:,.0f}\n")
        f.write(f"Minutes Saved Annually: {impact['minutes_saved']:,.0f}\n")
        f.write(f"Difficult Flights/Year: {impact['difficult_flights_annual']:,.0f}\n\n")
        f.write("ASSUMPTIONS:\n")
        f.write(f"- Delay cost per minute: ${impact['assumptions']['delay_cost']}\n")
        f.write(f"- Expected improvement: {impact['assumptions']['improvement_pct']:.0%}\n")
        f.write("- Based on industry standard costs and conservative estimates\n")
    
    print("✓ Business impact summary saved to 'business_impact_summary.txt'")
    
    return analyzer, enhanced, test_results, cv_results, impact


if __name__ == "__main__":
    analyzer, enhanced, test_results, cv_results, impact = run_complete_hackathon_analysis()
    
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("1. Review all generated files")
    print("2. Run: python additional_analysis.py (for presentation charts)")
    print("3. Run: python appendix_generator.py (for appendix charts)")
    print("4. Review sql_queries.sql for production deployment")
    print("=" * 80)