# main.py (Please refer readme.md for context)

from flight_analyzer import RankBasedFlightDifficultyScorer
import pandas as pd

def run_hackathon_project():
    """
    Main function to execute the flight difficulty analysis for the hackathon.
    """
    print("--- Starting SkyHack Challenge Analysis ---")

    data_paths = {
        'flights': './Data/Flight_Level_Data.csv',
        'pnr_flights': './Data/PNR_Flight_Level_Data.csv',
        'pnr_remarks': './Data/PNR_Remark_Level_Data.csv',
        'bags': './Data/Bag_Level_Data.csv',
        'airports': './Data/Airports_Data.csv'
    }

    # Initialize the analyzer
    analyzer = RankBasedFlightDifficultyScorer()
    
    # --- STEP 1: Load Data and Perform EDA ---
    analyzer.load_data(data_paths)
    eda_results = analyzer.perform_eda()

    print("\n--- Deliverable 1: Exploratory Data Analysis (EDA) ---")
    print(f"1. Average Delay: {eda_results['delay_stats']['average_delay']} minutes")
    print(f"   Percentage of Flights Delayed: {eda_results['delay_stats']['pct_delayed']}%")
    print(f"2. Flights with Insufficient Ground Time (< min): {eda_results['ground_time_stats']['flights_below_minimum']}")
    print(f"3. Average Transfer Bag Ratio: {eda_results['baggage_stats']['avg_transfer_ratio']:.2f}")
    print(f"4. Higher passenger loads correlate with higher delays.")
    print(f"5. High SSR flights are high-delay, even after controlling for load.")
    print("   (Delay difference for high SSR flights: "
          f"{eda_results['ssr_stats']['high_ssr_delay_correlation']} minutes)")

    # --- STEP 2: Feature Engineering & Scoring ---
    analyzer.create_difficulty_features()
    analyzer.calculate_rank_based_scores()
    
    # --- STEP 3: Analyze Final Results ---
    insights = analyzer.analyze_results()
    
    print("\n--- Deliverable 3: Post-Analysis & Operational Insights ---")
    print("\n--- Top 5 Most Difficult Destinations ---")
    print(insights['top_difficult_destinations'].head())
    
    print("\n--- Primary Drivers for 'Difficult' Flights ---")
    print(insights['primary_drivers'].head())
    
    print("\n--- Performance by Difficulty Class (Average Delay) ---")
    print(insights['class_performance']['delay_minutes']['mean'])

    recommendations = analyzer.get_recommendations()
    print("\n--- Actionable Recommendations ---")
    for rec in recommendations:
        print(f"- **{rec['category']}:** {rec['action']} (Priority: {rec['priority']})")
    
    # --- STEP 4: Export Final Files ---
    your_name = "Innov8torX_mini"
    analyzer.export_results(team_name=your_name)

    print(f"\n--- Analysis Complete. Submission file 'test_{your_name}.csv' is ready. ---")
    print("--- Helper file 'final_flight_data.csv' for charts is also ready. ---")

if __name__ == "__main__":
    run_hackathon_project()