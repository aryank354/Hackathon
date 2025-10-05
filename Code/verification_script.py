"""
Verification Script - Tests All Components Before Submission
Team: Innov8torX
Run this to ensure everything works before submitting
"""

import os
import sys
from pathlib import Path

def check_environment():
    """Check all dependencies are installed"""
    print("=" * 80)
    print("STEP 1: CHECKING ENVIRONMENT")
    print("=" * 80)
    
    required = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'matplotlib': 'Plotting',
        'seaborn': 'Statistical visualization',
        'sklearn': 'Machine learning'
    }
    
    missing = []
    for package, description in required.items():
        try:
            __import__(package)
            print(f"✓ {package:15s} - {description}")
        except ImportError:
            print(f"✗ {package:15s} - MISSING!")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Install missing packages:")
        if 'sklearn' in missing:
            print(f"   pip install scikit-learn")
        else:
            print(f"   pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All dependencies installed\n")
    return True


def check_data_files():
    """Check all required data files exist"""
    print("=" * 80)
    print("STEP 2: CHECKING DATA FILES")
    print("=" * 80)
    
    required_files = [
        '../Data/Flight_Level_Data.csv',
        '../Data/PNR_Flight_Level_Data.csv',
        '../Data/PNR_Remark_Level_Data.csv',
        '../Data/Bag_Level_Data.csv',
        '../Data/Airports_Data.csv'
    ]
    
    all_exist = True
    for filepath in required_files:
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✓ {filepath:50s} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {filepath:50s} - MISSING!")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Place all data files in ./Data/ directory")
        return False
    
    print("\n✓ All data files present\n")
    return True


def check_code_files():
    """Check all code files exist"""
    print("=" * 80)
    print("STEP 3: CHECKING CODE FILES")
    print("=" * 80)
    
    required_files = {
        'main.py': 'Original main script',
        'flight_analyzer.py': 'Base rank-based scorer',
        'enhanced_main.py': 'ML-enhanced main script',
        'enhanced_ml_analyzer.py': 'ML models & validation',
        'additional_analysis.py': 'Presentation charts',
        'appendix_generator.py': 'Appendix charts',
        'sql_queries.sql': 'Production SQL queries'
    }
    
    all_exist = True
    for filename, description in required_files.items():
        if os.path.exists(filename):
            print(f"✓ {filename:30s} - {description}")
        else:
            print(f"✗ {filename:30s} - MISSING!")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Create missing files from artifacts")
        return False
    
    print("\n✓ All code files present\n")
    return True


def run_quick_test():
    """Run a quick test of the base system"""
    print("=" * 80)
    print("STEP 4: RUNNING QUICK TEST")
    print("=" * 80)
    
    try:
        print("Testing base scorer import...")
        from flight_analyzer import RankBasedFlightDifficultyScorer
        print("✓ flight_analyzer.py imports successfully")
        
        print("\nTesting enhanced scorer import...")
        from enhanced_ml_analyzer import EnhancedMLFlightScorer
        print("✓ enhanced_ml_analyzer.py imports successfully")
        
        print("\n✓ All imports successful\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Import failed: {str(e)}\n")
        return False


def check_output_files():
    """Check if previous runs generated output"""
    print("=" * 80)
    print("STEP 5: CHECKING PREVIOUS OUTPUTS")
    print("=" * 80)
    
    output_files = {
        'test_Innov8torX_mini.csv': 'Submission file',
        'final_flight_data.csv': 'Complete dataset',
        'ml_validation_results.csv': 'ML validation (if enhanced_main ran)',
        'business_impact_summary.txt': 'ROI calculation (if enhanced_main ran)',
        'primary_drivers_chart.png': 'Chart 1',
        'top_destinations_chart.png': 'Chart 2',
        'delay_validation_chart.png': 'Chart 3'
    }

    for filename, description in output_files.items():
        # Check both root and Charts/ folder
        possible_paths = [filename, f"Charts/{filename}"]
        found = False
        for path in possible_paths:
            if os.path.exists(path):
                size_kb = os.path.getsize(path) / 1024
                print(f"✓ {path:40s} - {description} ({size_kb:.1f} KB)")
                found = True
                break
        if not found:
            print(f"  {filename:40s} - Not yet generated")
    
    print()


def generate_submission_checklist():
    """Print final submission checklist"""
    print("=" * 80)
    print("FINAL SUBMISSION CHECKLIST")
    print("=" * 80)
    
    checklist = [
        ("Files", [
            "test_Innov8torX_mini.csv (main submission)",
            "Presentation.pptx (15 slides)",
            "All Python code files",
            "sql_queries.sql",
            "All chart PNG files",
            "README.md with run instructions"
        ]),
        ("Code Quality", [
            "All code runs without errors",
            "enhanced_main.py generates ML results",
            "SQL queries are syntactically correct",
            "Comments explain key logic"
        ]),
        ("Presentation", [
            "15 slides maximum",
            "ML validation results shown (Slide 5)",
            "Business impact justified (Slide 8)",
            "SQL deployment mentioned (Slide 10)",
            "Validation pyramid (Slide 12)"
        ]),
        ("Validation", [
            "Temporal validation: Week 1→2",
            "Cross-validation: 5-fold",
            "Baseline comparisons included",
            "Confusion matrix shown"
        ]),
        ("Business Case", [
            "ROI: $1.8M - $2.5M justified",
            "Assumptions documented",
            "Sensitivity analysis shown",
            "Conservative estimates used"
        ])
    ]
    
    for category, items in checklist:
        print(f"\n{category}:")
        for item in items:
            print(f"  [ ] {item}")
    
    print("\n" + "=" * 80)


def main():
    """Run all verification steps"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "HACKATHON SUBMISSION VERIFICATION" + " " * 25 + "║")
    print("║" + " " * 25 + "Team: Innov8torX" + " " * 37 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    steps = [
        ("Environment", check_environment),
        ("Data Files", check_data_files),
        ("Code Files", check_code_files),
        ("Quick Test", run_quick_test)
    ]
    
    all_passed = True
    for step_name, step_func in steps:
        if not step_func():
            all_passed = False
            print(f"⚠️  {step_name} check FAILED")
            print("    Fix the issues above before proceeding\n")
    
    check_output_files()
    
    if all_passed:
        print("=" * 80)
        print("✓ ALL VERIFICATION CHECKS PASSED!")
        print("=" * 80)
        print("\nNext Steps:")
        print("1. Run: python enhanced_main.py")
        print("2. Run: python additional_analysis.py")
        print("3. Run: python appendix_generator.py")
        print("4. Create your 15-slide presentation")
        print("5. Review the final checklist below")
        print()
        generate_submission_checklist()
        print("\n✓ You're ready to win! 🏆\n")
    else:
        print("=" * 80)
        print("⚠️  SOME CHECKS FAILED")
        print("=" * 80)
        print("\nFix the issues above, then run this script again.")
        print()


if __name__ == "__main__":
    main()
