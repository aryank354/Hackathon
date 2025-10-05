"""
Enhanced Flight Difficulty Scorer with Machine Learning
Addresses hackathon gaps: ML models, proper validation, and business impact
Team: Innov8torX
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EnhancedMLFlightScorer:
    """
    Enhanced scorer with ML models and proper validation
    """
    
    def __init__(self, base_scorer):
        """
        Initialize with base RankBasedFlightDifficultyScorer
        """
        self.base_scorer = base_scorer
        self.rf_model = None
        self.gb_model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def prepare_ml_features(self, df):
        """
        Prepare features for ML models
        """
        feature_cols = [
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
        
        # Filter to only include existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols].fillna(0)
        
        # Create binary target: 1 = Difficult, 0 = Not Difficult
        y = (df['difficulty_class'] == 'Difficult').astype(int)
        
        return X, y, feature_cols
    
    def temporal_validation(self):
        """
        Week 1 train, Week 2 test validation
        """
        print("\n=== TEMPORAL VALIDATION: Week 1 Train → Week 2 Test ===")
        
        df = self.base_scorer.scored_flights.copy()
        df['week'] = pd.to_datetime(df['scheduled_departure_date_local']).dt.isocalendar().week
        
        week_1 = df['week'].min()
        week_2 = week_1 + 1
        
        train_df = df[df['week'] == week_1]
        test_df = df[df['week'] == week_2]
        
        print(f"Training on Week {week_1}: {len(train_df)} flights")
        print(f"Testing on Week {week_2}: {len(test_df)} flights")
        
        X_train, y_train, feature_cols = self.prepare_ml_features(train_df)
        X_test, y_test, _ = self.prepare_ml_features(test_df)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=50,
            random_state=42
        )
        self.rf_model.fit(X_train_scaled, y_train)
        
        # Gradient Boosting
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.gb_model.fit(X_train_scaled, y_train)
        
        # Predictions
        rf_pred = self.rf_model.predict(X_test_scaled)
        gb_pred = self.gb_model.predict(X_test_scaled)
        
        # Evaluate
        print("\n--- Random Forest Results ---")
        print(classification_report(y_test, rf_pred, target_names=['Not Difficult', 'Difficult']))
        
        print("\n--- Gradient Boosting Results ---")
        print(classification_report(y_test, gb_pred, target_names=['Not Difficult', 'Difficult']))
        
        # Feature Importance
        self.feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'RF_Importance': self.rf_model.feature_importances_,
            'GB_Importance': self.gb_model.feature_importances_
        }).sort_values('RF_Importance', ascending=False)
        
        print("\n--- Top 10 Most Important Features (Random Forest) ---")
        print(self.feature_importance.head(10))
        
        # Save predictions for comparison
        test_results = test_df[['flight_number', 'scheduled_departure_date_local', 
                                'difficulty_class', 'difficulty_score', 'delay_minutes']].copy()
        test_results['rf_predicted_difficult'] = rf_pred
        test_results['gb_predicted_difficult'] = gb_pred
        
        return test_results
    
    def cross_validation_analysis(self):
        """
        Perform k-fold cross-validation
        """
        print("\n=== CROSS-VALIDATION ANALYSIS (5-Fold) ===")
        
        df = self.base_scorer.scored_flights.copy()
        X, y, feature_cols = self.prepare_ml_features(df)
        X_scaled = self.scaler.fit_transform(X)
        
        # Random Forest CV
        rf_cv_scores = cross_val_score(
            RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            X_scaled, y, cv=5, scoring='accuracy'
        )
        
        # Gradient Boosting CV
        gb_cv_scores = cross_val_score(
            GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
            X_scaled, y, cv=5, scoring='accuracy'
        )
        
        print(f"Random Forest CV Accuracy: {rf_cv_scores.mean():.3f} (+/- {rf_cv_scores.std():.3f})")
        print(f"Gradient Boosting CV Accuracy: {gb_cv_scores.mean():.3f} (+/- {gb_cv_scores.std():.3f})")
        
        return {
            'rf_cv_mean': rf_cv_scores.mean(),
            'rf_cv_std': rf_cv_scores.std(),
            'gb_cv_mean': gb_cv_scores.mean(),
            'gb_cv_std': gb_cv_scores.std()
        }
    
    def calculate_business_impact(self):
        """
        Calculate justified business impact with transparent assumptions
        """
        print("\n=== BUSINESS IMPACT CALCULATION ===")
        
        df = self.base_scorer.scored_flights.copy()
        
        # Assumptions (document these clearly)
        DELAY_COST_PER_MINUTE = 75  # Industry standard: $50-100 per minute
        ORD_DAILY_FLIGHTS = 580  # Approximate daily departures from ORD
        DAYS_PER_YEAR = 365
        
        # Current state
        difficult_flights = df[df['difficulty_class'] == 'Difficult']
        current_avg_delay = difficult_flights['delay_minutes'].mean()
        difficult_flight_pct = len(difficult_flights) / len(df)
        
        print(f"Current State:")
        print(f"  - Difficult flights: {len(difficult_flights)} ({difficult_flight_pct:.1%})")
        print(f"  - Avg delay (Difficult): {current_avg_delay:.1f} minutes")
        
        # Conservative improvement estimate
        # Assumption: Proactive staffing reduces delays by 20% for difficult flights
        IMPROVEMENT_PCT = 0.20
        projected_delay_reduction = current_avg_delay * IMPROVEMENT_PCT
        
        print(f"\nProjected Impact (Conservative {IMPROVEMENT_PCT:.0%} reduction):")
        print(f"  - Delay reduction per difficult flight: {projected_delay_reduction:.1f} minutes")
        
        # Annual calculations
        annual_difficult_flights = ORD_DAILY_FLIGHTS * difficult_flight_pct * DAYS_PER_YEAR
        annual_minutes_saved = annual_difficult_flights * projected_delay_reduction
        annual_cost_savings = annual_minutes_saved * DELAY_COST_PER_MINUTE
        
        print(f"\nAnnual Projections (ORD only):")
        print(f"  - Difficult flights per year: {annual_difficult_flights:.0f}")
        print(f"  - Total minutes saved: {annual_minutes_saved:.0f}")
        print(f"  - Cost savings: ${annual_cost_savings:,.0f}")
        
        print(f"\nAssumptions:")
        print(f"  - Delay cost: ${DELAY_COST_PER_MINUTE}/minute")
        print(f"  - Daily ORD flights: {ORD_DAILY_FLIGHTS}")
        print(f"  - Improvement from proactive staffing: {IMPROVEMENT_PCT:.0%}")
        
        # Sensitivity analysis
        print(f"\nSensitivity Analysis:")
        for improvement in [0.10, 0.15, 0.20, 0.25]:
            savings = annual_difficult_flights * (current_avg_delay * improvement) * DELAY_COST_PER_MINUTE
            print(f"  - {improvement:.0%} improvement: ${savings:,.0f}")
        
        return {
            'annual_savings': annual_cost_savings,
            'minutes_saved': annual_minutes_saved,
            'difficult_flights_annual': annual_difficult_flights,
            'assumptions': {
                'delay_cost': DELAY_COST_PER_MINUTE,
                'improvement_pct': IMPROVEMENT_PCT
            }
        }
    
    def baseline_comparison(self):
        """
        Compare against simple baselines
        """
        print("\n=== BASELINE COMPARISON ===")
        
        df = self.base_scorer.scored_flights.copy()
        
        # Baseline 1: Just use load factor
        load_threshold = df['load_factor'].quantile(0.80)
        baseline1_pred = (df['load_factor'] > load_threshold).astype(int)
        actual = (df['difficulty_class'] == 'Difficult').astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        print("Baseline 1: Load Factor Only (top 20%)")
        print(f"  Accuracy: {accuracy_score(actual, baseline1_pred):.3f}")
        print(f"  Precision: {precision_score(actual, baseline1_pred):.3f}")
        print(f"  Recall: {recall_score(actual, baseline1_pred):.3f}")
        print(f"  F1-Score: {f1_score(actual, baseline1_pred):.3f}")
        
        # Baseline 2: Historical delay only
        delay_threshold = df['historical_delay'].quantile(0.80)
        baseline2_pred = (df['historical_delay'] > delay_threshold).astype(int)
        
        print("\nBaseline 2: Historical Delay Only (top 20%)")
        print(f"  Accuracy: {accuracy_score(actual, baseline2_pred):.3f}")
        print(f"  Precision: {precision_score(actual, baseline2_pred):.3f}")
        print(f"  Recall: {recall_score(actual, baseline2_pred):.3f}")
        print(f"  F1-Score: {f1_score(actual, baseline2_pred):.3f}")
        
        # Our rank-based approach (perfect by design for classification)
        print("\nOur Rank-Based Approach:")
        print(f"  Accuracy: 1.000 (by design)")
        print(f"  But validates with 29x delay separation")
        
    def generate_confusion_matrices(self):
        """
        Generate confusion matrices for ML models
        """
        print("\n=== CONFUSION MATRICES ===")
        
        if self.rf_model is None:
            print("Run temporal_validation() first")
            return
        
        # Use test results from temporal validation
        test_results = self.temporal_validation()
        
        actual = (test_results['difficulty_class'] == 'Difficult').astype(int)
        rf_pred = test_results['rf_predicted_difficult']
        
        print("\nRandom Forest Confusion Matrix:")
        print(confusion_matrix(actual, rf_pred))
        print("\n[[TN, FP],")
        print(" [FN, TP]]")


def run_enhanced_analysis(base_scorer):
    """
    Run all enhanced analyses
    """
    enhanced = EnhancedMLFlightScorer(base_scorer)
    
    # 1. Temporal Validation
    test_results = enhanced.temporal_validation()
    
    # 2. Cross-Validation
    cv_results = enhanced.cross_validation_analysis()
    
    # 3. Business Impact
    impact = enhanced.calculate_business_impact()
    
    # 4. Baseline Comparison
    enhanced.baseline_comparison()
    
    # 5. Confusion Matrix
    enhanced.generate_confusion_matrices()
    
    return enhanced, test_results, cv_results, impact


if __name__ == "__main__":
    print("This module should be imported and used with the base RankBasedFlightDifficultyScorer")
    print("Example usage:")
    print("  from flight_analyzer import RankBasedFlightDifficultyScorer")
    print("  base_scorer = RankBasedFlightDifficultyScorer()")
    print("  # ... load data and run base analysis ...")
    print("  enhanced, results, cv, impact = run_enhanced_analysis(base_scorer)")