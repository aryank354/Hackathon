"""
United Airlines Flight Difficulty Score System - Robust Rank-Based Approach with Strong Features and EDA
Team: Innov8torX_mini 
Member: Aryan Kanojia, Anurag Kumar
Date: 04-10-2025
Description: Our monster Rank-based scoring system for flight difficulty without complex weights
Output file for submission: test_Innov8torX_mini.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class RankBasedFlightDifficultyScorer:
    """
    Rank-based flight difficulty scoring system
    Uses percentile ranking to avoid scaling and weighting complexities
    """
    
    def __init__(self):
        """
        Initialize the rank-based scorer
        """
        self.logger = logging.getLogger(__name__)
        self.data_loaded = False
        self.features_created = False
        self.scores_calculated = False
        
    def load_data(self, data_paths):
        """
        Load all required datasets
        
        Parameters:
        -----------
        data_paths : dict
            Dictionary with paths to each CSV file
            Keys: 'flights', 'pnr_flights', 'pnr_remarks', 'bags', 'airports'
        """
        self.logger.info("Loading datasets...")
        
        try:
            # Load flight level data
            self.flights_df = pd.read_csv(data_paths['flights'])
            self.logger.info(f"Loaded {len(self.flights_df)} flight records")
            
            # Load PNR flight level data
            self.pnr_flights_df = pd.read_csv(data_paths['pnr_flights'])
            self.logger.info(f"Loaded {len(self.pnr_flights_df)} PNR flight records")
            
            # Load PNR remarks (special service requests)
            self.pnr_remarks_df = pd.read_csv(data_paths['pnr_remarks'])
            self.logger.info(f"Loaded {len(self.pnr_remarks_df)} PNR remarks")
            
            # Load bag level data
            self.bags_df = pd.read_csv(data_paths['bags'])
            self.logger.info(f"Loaded {len(self.bags_df)} bag records")
            
            # Load airports data
            self.airports_df = pd.read_csv(data_paths['airports'])
            self.logger.info(f"Loaded {len(self.airports_df)} airport records")
            
            # Convert datetime columns
            self._convert_datetime_columns()
            
            self.data_loaded = True
            return self
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _convert_datetime_columns(self):
        """
        Convert string datetime columns to datetime objects
        """
        # Flight datetime columns
        datetime_cols = [
            'scheduled_departure_datetime_local',
            'scheduled_arrival_datetime_local', 
            'actual_departure_datetime_local',
            'actual_arrival_datetime_local'
        ]
        
        for col in datetime_cols:
            if col in self.flights_df.columns:
                self.flights_df[col] = pd.to_datetime(self.flights_df[col], errors='coerce')
        
        # Convert date columns
        self.flights_df['scheduled_departure_date_local'] = pd.to_datetime(
            self.flights_df['scheduled_departure_date_local']
        )
        
        # PNR creation date
        if 'pnr_creation_date' in self.pnr_flights_df.columns:
            self.pnr_flights_df['pnr_creation_date'] = pd.to_datetime(
                self.pnr_flights_df['pnr_creation_date'], errors='coerce'
            )
        
        # Bag tag issue date
        if 'bag_tag_issue_date' in self.bags_df.columns:
            self.bags_df['bag_tag_issue_date'] = pd.to_datetime(
                self.bags_df['bag_tag_issue_date'], errors='coerce'
            )
    
    def perform_eda(self, save_plots=False):
        """
        Perform Exploratory Data Analysis
        Returns dict with key statistics
        """
        if not self.data_loaded:
            raise ValueError("Data not loaded. Please run load_data() first.")
        
        self.logger.info("Performing Exploratory Data Analysis...")
        
        eda_results = {}
        
        # 1. DELAY ANALYSIS
        self.logger.info("Analyzing delays...")
        self.flights_df['delay_minutes'] = (
            (self.flights_df['actual_departure_datetime_local'] - 
             self.flights_df['scheduled_departure_datetime_local']).dt.total_seconds() / 60
        )
        
        eda_results['delay_stats'] = {
            'average_delay': round(self.flights_df['delay_minutes'].mean(), 2),
            'median_delay': round(self.flights_df['delay_minutes'].median(), 2),
            'std_delay': round(self.flights_df['delay_minutes'].std(), 2),
            'pct_delayed': round((self.flights_df['delay_minutes'] > 0).mean() * 100, 2),
            'pct_delayed_15min': round((self.flights_df['delay_minutes'] > 15).mean() * 100, 2)
        }
        
        self.logger.info(f"Average delay: {eda_results['delay_stats']['average_delay']} minutes")
        self.logger.info(f"% Flights delayed: {eda_results['delay_stats']['pct_delayed']}%")
        
        # 2. GROUND TIME ANALYSIS
        self.logger.info("Analyzing ground time...")
        self.flights_df['ground_time_buffer'] = (
            self.flights_df['scheduled_ground_time_minutes'] - 
            self.flights_df['minimum_turn_minutes']
        )
        
        eda_results['ground_time_stats'] = {
            'flights_below_minimum': (self.flights_df['ground_time_buffer'] < 0).sum(),
            'pct_below_minimum': round((self.flights_df['ground_time_buffer'] < 0).mean() * 100, 2),
            'flights_tight_turn': (self.flights_df['ground_time_buffer'] <= 5).sum(),
            'pct_tight_turn': round((self.flights_df['ground_time_buffer'] <= 5).mean() * 100, 2),
            'avg_buffer': round(self.flights_df['ground_time_buffer'].mean(), 2)
        }
        
        self.logger.info(f"Flights below minimum turn time: {eda_results['ground_time_stats']['flights_below_minimum']}")
        
        # 3. BAGGAGE ANALYSIS
        self.logger.info("Analyzing baggage...")
        bag_summary = self.bags_df.groupby(
            ['company_id', 'flight_number', 'scheduled_departure_date_local']
        ).agg({
            'bag_tag_unique_number': 'count',
            'bag_type': lambda x: (x == 'Transfer').sum()
        }).rename(columns={
            'bag_tag_unique_number': 'total_bags',
            'bag_type': 'transfer_bags'
        })
        
        bag_summary['transfer_ratio'] = bag_summary['transfer_bags'] / (bag_summary['total_bags'] + 1)
        bag_summary['checked_bags'] = bag_summary['total_bags'] - bag_summary['transfer_bags']
        
        eda_results['baggage_stats'] = {
            'avg_bags_per_flight': round(bag_summary['total_bags'].mean(), 2),
            'avg_transfer_ratio': round(bag_summary['transfer_ratio'].mean(), 3),
            'avg_checked_vs_transfer': round(
                bag_summary['checked_bags'].sum() / (bag_summary['transfer_bags'].sum() + 1), 2
            )
        }
        
        self.logger.info(f"Average transfer bag ratio: {eda_results['baggage_stats']['avg_transfer_ratio']:.1%}")
        
        # 4. PASSENGER LOAD ANALYSIS
        self.logger.info("Analyzing passenger loads...")
        self.pnr_flights_df['scheduled_departure_date_local'] = pd.to_datetime(self.pnr_flights_df['scheduled_departure_date_local'])

        pnr_summary = self.pnr_flights_df.groupby(
            ['company_id', 'flight_number', 'scheduled_departure_date_local']
        ).agg({
            'total_pax': 'sum',
            'lap_child_count': 'sum',
            'basic_economy_ind': 'sum'
        })
        
        # Merge with flights for load factor
        flight_pax = self.flights_df.merge(
            pnr_summary,
            on=['company_id', 'flight_number', 'scheduled_departure_date_local'],
            how='left'
        )
        flight_pax['load_factor'] = flight_pax['total_pax'] / (flight_pax['total_seats'] + 1)
        
        eda_results['passenger_stats'] = {
            'avg_load_factor': round(flight_pax['load_factor'].mean(), 3),
            'flights_above_85pct': (flight_pax['load_factor'] > 0.85).sum(),
            'flights_above_95pct': (flight_pax['load_factor'] > 0.95).sum(),
            'avg_pax_per_flight': round(flight_pax['total_pax'].mean(), 1)
        }
        
        self.logger.info(f"Average load factor: {eda_results['passenger_stats']['avg_load_factor']:.1%}")
        
        # 5. SPECIAL SERVICE ANALYSIS
        self.logger.info("Analyzing special service requests...")
        
        # Count SSRs per flight
        ssr_counts = self.pnr_remarks_df.groupby('flight_number').size()
        
        # Merge SSR counts with flight_pax to get SSR per passenger
        flight_ssr = flight_pax.merge(
            ssr_counts.rename('total_ssr').reset_index(),
            on='flight_number',
            how='left'
        )
        flight_ssr['total_ssr'].fillna(0, inplace=True)
        flight_ssr['ssr_per_pax'] = flight_ssr['total_ssr'] / (flight_ssr['total_pax'] + 1)
        
        # Check correlation between SSR and delays (controlling for load)
        high_ssr_mask = flight_ssr['ssr_per_pax'] > flight_ssr['ssr_per_pax'].quantile(0.75)
        high_load_mask = flight_ssr['load_factor'] > 0.85
        
        eda_results['ssr_stats'] = {
            'avg_ssr_per_flight': round(ssr_counts.mean(), 2),
            'max_ssr_on_flight': int(ssr_counts.max()),
            'flights_with_ssr': len(ssr_counts),
            'high_ssr_delay_correlation': round(
                flight_ssr[high_ssr_mask]['delay_minutes'].mean() - 
                flight_ssr[~high_ssr_mask]['delay_minutes'].mean(), 2
            ),
            'high_ssr_high_load_delays': round(
                flight_ssr[high_ssr_mask & high_load_mask]['delay_minutes'].mean(), 2
            ) if any(high_ssr_mask & high_load_mask) else 0
        }
        
        self.logger.info(f"Flights with high SSR average {eda_results['ssr_stats']['high_ssr_delay_correlation']} min more delay")
        
        self.eda_results = eda_results
        return eda_results
    
    def create_difficulty_features(self):
        """
        Create all features that will be used for ranking
        Each feature represents a difficulty dimension
        """
        if not self.data_loaded:
            raise ValueError("Data not loaded. Please run load_data() first.")
        
        self.logger.info("Creating difficulty features...")
        
        # Start with base flight data
        df = self.flights_df.copy()
        
        # Create flight key for consistent joining
        df['flight_key'] = (
            df['company_id'] + '_' + 
            df['flight_number'].astype(str) + '_' + 
            df['scheduled_departure_date_local'].astype(str)
        )
        
        # 1. GROUND TIME PRESSURE
        df['ground_time_pressure'] = np.where(
            df['scheduled_ground_time_minutes'] > 0,
            df['minimum_turn_minutes'] / df['scheduled_ground_time_minutes'],
            2.0  # High pressure if no ground time
        ).clip(0, 2)
        
        df['insufficient_ground_time'] = (
            df['scheduled_ground_time_minutes'] < df['minimum_turn_minutes']
        ).astype(int)
        
        # 2. PASSENGER COMPLEXITY
        self.pnr_flights_df['scheduled_departure_date_local'] = pd.to_datetime(
            self.pnr_flights_df['scheduled_departure_date_local']
        )
        # Aggregate passenger data
        pnr_agg = self.pnr_flights_df.groupby(
            ['company_id', 'flight_number', 'scheduled_departure_date_local']
        ).agg({
            'total_pax': 'sum',
            'lap_child_count': 'sum',
            'is_child': lambda x: (x == 'Y').sum(),
            'basic_economy_ind': 'sum',
            'is_stroller_user': lambda x: (x == 'Y').sum()
        }).reset_index()
        
        df = df.merge(pnr_agg, 
                     on=['company_id', 'flight_number', 'scheduled_departure_date_local'],
                     how='left')
        
        # Fill missing values
        pax_cols = ['total_pax', 'lap_child_count', 'is_child', 'basic_economy_ind', 'is_stroller_user']
        df[pax_cols] = df[pax_cols].fillna(0)
        
        # Calculate load factor
        df['load_factor'] = df['total_pax'] / (df['total_seats'] + 1)
        
        # Family travel complexity (more children = more complexity)
        df['family_complexity'] = (
            df['lap_child_count'] + 
            df['is_child'] * 0.5 + 
            df['is_stroller_user'] * 0.7
        )
        
        # 3. BAGGAGE COMPLEXITY
        self.bags_df['scheduled_departure_date_local'] = pd.to_datetime(self.bags_df['scheduled_departure_date_local'])

        bag_agg = self.bags_df.groupby(
            ['company_id', 'flight_number', 'scheduled_departure_date_local']
        ).agg({
            'bag_tag_unique_number': 'count',
            'bag_type': lambda x: (x == 'Transfer').sum()
        }).rename(columns={
            'bag_tag_unique_number': 'total_bags',
            'bag_type': 'transfer_bags'
        }).reset_index()
        
        df = df.merge(bag_agg,
                     on=['company_id', 'flight_number', 'scheduled_departure_date_local'],
                     how='left')
        
        df[['total_bags', 'transfer_bags']] = df[['total_bags', 'transfer_bags']].fillna(0)
        
        # Bags per passenger (handling complexity)
        df['bags_per_pax'] = df['total_bags'] / (df['total_pax'] + 1)
        
        # Transfer bag percentage (requires special handling)
        df['transfer_bag_pct'] = df['transfer_bags'] / (df['total_bags'] + 1)
        
        # 4. SPECIAL SERVICE REQUIREMENTS
        ssr_counts = self.pnr_remarks_df.groupby('flight_number').agg({
            'special_service_request': 'count'
        }).rename(columns={'special_service_request': 'total_ssr'}).reset_index()
        
        df = df.merge(ssr_counts, on='flight_number', how='left')
        df['total_ssr'] = df['total_ssr'].fillna(0)
        
        # SSR intensity
        df['ssr_intensity'] = df['total_ssr'] / (df['total_pax'] + 1)
        
        # 5. OPERATIONAL PERFORMANCE
        df['delay_minutes'] = (
            (df['actual_departure_datetime_local'] - 
             df['scheduled_departure_datetime_local']).dt.total_seconds() / 60
        )
        
        # Historical delay tendency (will be used as a feature)
        df['historical_delay'] = df['delay_minutes'].fillna(0)
        
        # 6. FLIGHT CHARACTERISTICS
        # Time of day
        df['hour'] = df['scheduled_departure_datetime_local'].dt.hour
        df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if (6 <= x <= 9) or (16 <= x <= 20) else 0)
        
        # Aircraft size impact
        df['aircraft_size'] = pd.cut(
            df['total_seats'],
            bins=[0, 50, 150, 300, 1000],
            labels=[1, 2, 3, 4]
        ).fillna(2).astype(int)
        
        # International flights (using airport data)
        international = self.airports_df[
            self.airports_df['iso_country_code'] != 'US'
        ]['airport_iata_code'].unique()
        
        df['is_international'] = df['scheduled_arrival_station_code'].isin(international).astype(int)
        
        # 7. COMPOSITE DIFFICULTY INDICATORS
        # High stress flight (multiple difficulty factors)
        df['high_stress_flight'] = (
            (df['load_factor'] > 0.85) & 
            (df['ground_time_pressure'] > 1) & 
            (df['is_peak_hour'] == 1)
        ).astype(int)
        
        # Store the features dataframe
        self.features_df = df
        self.features_created = True
        
        self.logger.info(f"Created {len(df.columns)} total features for {len(df)} flights")
        
        return df
    
    def calculate_rank_based_scores(self):
        """
        Calculate difficulty scores using rank-based approach
        No weights, no scaling - pure percentile ranking
        """
        if not self.features_created:
            raise ValueError("Features not created. Please run create_difficulty_features() first.")
        
        self.logger.info("Calculating rank-based difficulty scores...")
        
        df = self.features_df.copy()
        
        # Define difficulty indicators (higher value = more difficult)
        difficulty_features = [
            'ground_time_pressure',
            'insufficient_ground_time',
            'load_factor',
            'family_complexity',
            'bags_per_pax',
            'transfer_bag_pct',
            'ssr_intensity',
            'historical_delay',
            'is_peak_hour',
            'aircraft_size',
            'is_international',
            'high_stress_flight'
        ]
        
        # Calculate daily percentile ranks for each feature
        for feature in difficulty_features:
            if feature in df.columns:
                df[f'{feature}_rank'] = df.groupby('scheduled_departure_date_local')[feature].rank(
                    method='average',
                    ascending=True,
                    pct=True
                )
                df[f'{feature}_rank'].fillna(0.5, inplace=True)
        
        # Calculate composite difficulty score as AVERAGE of all ranks
        rank_columns = [f'{feat}_rank' for feat in difficulty_features if f'{feat}_rank' in df.columns]
        df['difficulty_score_raw'] = df[rank_columns].mean(axis=1)
        
        # Convert to 0-100 scale
        df['difficulty_score'] = df['difficulty_score_raw'] * 100
        
        # Calculate daily rank (1 = most difficult)
        df['daily_rank'] = df.groupby('scheduled_departure_date_local')['difficulty_score'].rank(
            method='dense',
            ascending=False
        )
        
        # Calculate percentile within each day
        df['daily_percentile'] = df.groupby('scheduled_departure_date_local')['difficulty_score'].rank(
            method='average',
            ascending=False,
            pct=True
        )
        
        # Classify based on percentiles
        def classify_difficulty(percentile):
            if percentile <= 0.20:
                return 'Difficult'
            elif percentile <= 0.70:
                return 'Medium'
            else:
                return 'Easy'
        
        df['difficulty_class'] = df['daily_percentile'].apply(classify_difficulty)
        
        # Identify primary difficulty driver
        driver_ranks = df[rank_columns]
        df['primary_driver'] = driver_ranks.idxmax(axis=1).str.replace('_rank', '')
        
        # Calculate confidence based on data completeness
        data_features = ['total_pax', 'total_bags', 'total_ssr', 'delay_minutes']
        df['data_completeness'] = df[data_features].notna().mean(axis=1)
        
        self.scored_flights = df
        self.scores_calculated = True
        
        self.logger.info("Rank-based scoring complete")
        self.logger.info(f"Difficult flights: {(df['difficulty_class'] == 'Difficult').sum()}")
        self.logger.info(f"Medium flights: {(df['difficulty_class'] == 'Medium').sum()}")
        self.logger.info(f"Easy flights: {(df['difficulty_class'] == 'Easy').sum()}")
        
        return df
    
    def analyze_results(self):
        """
        Analyze results and generate operational insights
        """
        if not self.scores_calculated:
            raise ValueError("Scores not calculated. Please run calculate_rank_based_scores() first.")
        
        self.logger.info("Analyzing results...")
        
        df = self.scored_flights.copy()
        insights = {}
        
        # 1. DESTINATION DIFFICULTY ANALYSIS
        destination_stats = df.groupby('scheduled_arrival_station_code').agg({
            'difficulty_score': ['mean', 'std', 'count'],
            'difficulty_class': lambda x: (x == 'Difficult').mean(),
            'delay_minutes': 'mean',
            'load_factor': 'mean',
            'ground_time_pressure': 'mean'
        }).round(2)
        
        destination_stats.columns = ['_'.join(col).strip() for col in destination_stats.columns]
        destination_stats = destination_stats.sort_values('difficulty_score_mean', ascending=False)
        
        insights['top_difficult_destinations'] = destination_stats.head(10)
        
        # 2. COMMON DRIVERS FOR DIFFICULT FLIGHTS
        difficult_flights = df[df['difficulty_class'] == 'Difficult']
        
        driver_summary = {
            'Ground Time Issues': (difficult_flights['insufficient_ground_time'] == 1).mean(),
            'High Load (>85%)': (difficult_flights['load_factor'] > 0.85).mean(),
            'Peak Hour Operations': (difficult_flights['is_peak_hour'] == 1).mean(),
            'High Baggage Volume': (difficult_flights['bags_per_pax'] > 
                                   df['bags_per_pax'].quantile(0.75)).mean(),
            'Transfer Bag Complexity': (difficult_flights['transfer_bag_pct'] > 0.2).mean(),
            'Special Service Intensity': (difficult_flights['ssr_intensity'] > 
                                         df['ssr_intensity'].quantile(0.75)).mean(),
            'International Flights': (difficult_flights['is_international'] == 1).mean(),
            'Large Aircraft': (difficult_flights['aircraft_size'] >= 3).mean()
        }
        
        insights['difficulty_drivers'] = pd.Series(driver_summary).sort_values(ascending=False)
        
        # 3. PRIMARY DRIVER DISTRIBUTION
        primary_driver_dist = difficult_flights['primary_driver'].value_counts()
        insights['primary_drivers'] = primary_driver_dist
        
        # 4. TIME PATTERN ANALYSIS
        hourly_difficulty = df.groupby('hour').agg({
            'difficulty_score': 'mean',
            'difficulty_class': lambda x: (x == 'Difficult').mean()
        })
        
        insights['hourly_patterns'] = hourly_difficulty
        
        # 5. OPERATIONAL IMPACT
        class_performance = df.groupby('difficulty_class').agg({
            'delay_minutes': ['mean', 'std'],
            'daily_rank': 'count',
            'load_factor': 'mean',
            'ground_time_pressure': 'mean'
        }).round(2)
        
        insights['class_performance'] = class_performance
        
        self.insights = insights
        return insights
    
    def get_recommendations(self):
        """
        Generate specific operational recommendations based on analysis
        """
        if not hasattr(self, 'insights'):
            self.analyze_results()
        
        self.logger.info("Generating recommendations...")
        
        df = self.scored_flights
        insights = self.insights
        
        recommendations = []
        
        # 1. GROUND OPERATIONS
        ground_time_issues = (df['insufficient_ground_time'] == 1).mean()
        if ground_time_issues > 0.1:
            recommendations.append({
                'category': 'Ground Operations',
                'issue': f'{ground_time_issues:.1%} of flights have insufficient ground time',
                'action': 'Pre-position ground crews and equipment for these flights',
                'priority': 'High',
                'affected_flights': df[df['insufficient_ground_time'] == 1]['flight_number'].nunique()
            })
        
        # 2. PEAK HOUR MANAGEMENT
        peak_difficult = df[df['is_peak_hour'] == 1]['difficulty_class'].value_counts(normalize=True)
        if 'Difficult' in peak_difficult and peak_difficult['Difficult'] > 0.3:
            recommendations.append({
                'category': 'Peak Hour Operations',
                'issue': f'{peak_difficult["Difficult"]:.1%} of peak hour flights are difficult',
                'action': 'Increase staffing during 6-9 AM and 4-8 PM periods',
                'priority': 'High',
                'affected_flights': df[(df['is_peak_hour'] == 1) & 
                                      (df['difficulty_class'] == 'Difficult')]['flight_number'].nunique()
            })
        
        # 3. HIGH LOAD FLIGHTS
        high_load_difficult = df[df['load_factor'] > 0.85]
        if len(high_load_difficult) > 0:
            avg_delay = high_load_difficult['delay_minutes'].mean()
            recommendations.append({
                'category': 'Passenger Management',
                'issue': f'High load flights average {avg_delay:.1f} min delay',
                'action': 'Open additional check-in counters and boarding gates early',
                'priority': 'Medium',
                'affected_flights': high_load_difficult['flight_number'].nunique()
            })
        
        # 4. SPECIAL SERVICE REQUIREMENTS
        high_ssr = df[df['ssr_intensity'] > df['ssr_intensity'].quantile(0.75)]
        if len(high_ssr) > 0:
            recommendations.append({
                'category': 'Special Services',
                'issue': f'{len(high_ssr)} flights have high special service requirements',
                'action': 'Assign dedicated assistance team members proactively',
                'priority': 'Medium',
                'affected_flights': high_ssr['flight_number'].nunique()
            })
        
        # 5. TRANSFER BAGGAGE
        high_transfer = df[df['transfer_bag_pct'] > 0.3]
        if len(high_transfer) > 0:
            recommendations.append({
                'category': 'Baggage Handling',
                'issue': f'{len(high_transfer)} flights have >30% transfer bags',
                'action': 'Prioritize baggage carts and handlers for transfer-heavy flights',
                'priority': 'High',
                'affected_flights': high_transfer['flight_number'].nunique()
            })
        
        # 6. DESTINATION-SPECIFIC
        top_difficult = insights['top_difficult_destinations'].head(3)
        for destination in top_difficult.index:
            recommendations.append({
                'category': 'Route-Specific',
                'issue': f'{destination} routes consistently show high difficulty',
                'action': f'Create specialized handling procedures for {destination} flights',
                'priority': 'Medium',
                'affected_flights': df[df['scheduled_arrival_station_code'] == destination]['flight_number'].nunique()
            })
        
        return recommendations
    
    def export_results(self, team_name='Innov8torX_mini', output_path=None):
        """
        Export results to CSV file for submission
        """
        if not self.scores_calculated:
            raise ValueError("Scores not calculated. Please run calculate_rank_based_scores() first.")
        
        if output_path is None:
            output_path = f'test_{team_name}.csv'
        
        # Main submission file columns
        submission_columns = [
            'company_id', 'flight_number', 'scheduled_departure_date_local',
            'scheduled_departure_station_code', 'scheduled_arrival_station_code',
            'scheduled_departure_datetime_local',
            'ground_time_pressure', 'insufficient_ground_time', 'load_factor',
            'family_complexity', 'bags_per_pax', 'transfer_bag_pct',
            'ssr_intensity', 'historical_delay', 'is_peak_hour',
            'aircraft_size', 'is_international', 'high_stress_flight',
            'difficulty_score', 'daily_rank', 'daily_percentile',
            'difficulty_class', 'primary_driver'
        ]
        
        output_df = self.scored_flights[submission_columns].copy()
        output_df = output_df.sort_values(['scheduled_departure_date_local', 'daily_rank'])
        output_df.to_csv(output_path, index=False)
        self.logger.info(f"✅ SUBMISSION FILE exported to '{output_path}'")
        
        # Helper file with all columns
        full_output_columns = [
            'company_id', 'flight_number', 'scheduled_departure_date_local',
            'scheduled_departure_station_code', 'scheduled_arrival_station_code',
            'scheduled_departure_datetime_local', 'scheduled_arrival_datetime_local',
            'actual_departure_datetime_local', 'actual_arrival_datetime_local',
            'delay_minutes', 'total_seats', 'fleet_type', 'carrier',
            'scheduled_ground_time_minutes', 'minimum_turn_minutes', 'hour',
            'ground_time_pressure', 'insufficient_ground_time', 'load_factor',
            'family_complexity', 'bags_per_pax', 'transfer_bag_pct',
            'ssr_intensity', 'historical_delay', 'is_peak_hour',
            'aircraft_size', 'is_international', 'high_stress_flight',
            'total_pax', 'lap_child_count', 'is_child', 'basic_economy_ind',
            'is_stroller_user', 'total_bags', 'transfer_bags', 'total_ssr',
            'difficulty_score', 'daily_rank', 'daily_percentile',
            'difficulty_class', 'primary_driver', 'data_completeness'
        ]
        
        # Only include columns that exist
        full_output_columns = [col for col in full_output_columns if col in self.scored_flights.columns]
        
        full_output_df = self.scored_flights[full_output_columns].copy()
        full_output_df = full_output_df.sort_values(['scheduled_departure_date_local', 'daily_rank'])
        
        helper_path = 'final_flight_data.csv'
        full_output_df.to_csv(helper_path, index=False)
        self.logger.info(f"✅ HELPER FILE exported to '{helper_path}' (for appendix charts)")
        
        return output_df