import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_final_visualizations(data_file='final_flight_data.csv'):
    """
    Creates three final, presentation-ready charts based on the successful analysis.
    """
    print("\n--- Generating Final Presentation Visuals ---")
    try:
        df = pd.read_csv(data_file)
        print(f"✅ Successfully loaded '{data_file}' for visualization.")
    except FileNotFoundError:
        print(f"ERROR: '{data_file}' not found. Please run 'main.py' first to generate it.")
        return

    # Ensure Charts folder exists
    charts_dir = os.path.join(os.getcwd(), "Charts")
    os.makedirs(charts_dir, exist_ok=True)

    plt.style.use('seaborn-v0_8-talk')

    # === Chart 1: The "Why" Chart (Primary Drivers) ===
    difficult_flights = df[df['difficulty_class'] == 'Difficult']
    primary_drivers = difficult_flights['primary_driver'].value_counts().head(5)

    plt.figure(figsize=(14, 8))
    sns.barplot(x=primary_drivers.values, y=primary_drivers.index, palette='rocket', orient='h')
    plt.title("Top 5 Primary Drivers of 'Difficult' Flights", fontsize=20, weight='bold')
    plt.xlabel("Number of Times as #1 Cause", fontsize=14)
    plt.ylabel("Primary Driver Feature", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'primary_drivers_chart.png'))
    print("Chart 1: 'primary_drivers_chart.png' saved in Charts/")

    # === Chart 2: The "Where" Chart (Difficult Destinations) ===
    top_destinations = difficult_flights['scheduled_arrival_station_code'].value_counts().head(10)

    plt.figure(figsize=(14, 8))
    sns.barplot(x=top_destinations.index, y=top_destinations.values, palette='viridis')
    plt.title("Top 10 Most Frequent 'Difficult' Destinations", fontsize=20, weight='bold')
    plt.ylabel("Number of 'Difficult' Flights", fontsize=14)
    plt.xlabel("Destination Airport", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'top_destinations_chart.png'))
    print("Chart 2: 'top_destinations_chart.png' saved in Charts/")

    # === Chart 3: The "Proof" Chart (Validation) ===
    performance = df.groupby('difficulty_class')['delay_minutes'].mean().reindex(['Easy', 'Medium', 'Difficult'])

    plt.figure(figsize=(10, 7))
    sns.barplot(x=performance.index, y=performance.values, palette='coolwarm')
    plt.title("Average Delay by Difficulty Class", fontsize=20, weight='bold')
    plt.ylabel("Average Departure Delay (Minutes)", fontsize=14)
    plt.xlabel("Calculated Difficulty Class", fontsize=14)
    for index, value in enumerate(performance.values):
        plt.text(index, value + 1, f'{value:.1f} min', ha='center', va='bottom', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'delay_validation_chart.png'))
    print("Chart 3: 'delay_validation_chart.png' saved in Charts/")

    plt.close('all')

if __name__ == "__main__":
    create_final_visualizations()
    print("\n🎉 All 3 presentation charts have been successfully created in the 'Charts' folder!")
