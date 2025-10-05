# SkyHack Challenge: Flight Difficulty Score

Github Repository: https://github.com/aryank354/Hackathon

## 1. Project Overview

This project delivers a complete, end-to-end solution for the SkyHack Challenge. Our goal was to create a systematic, data-driven Flight Difficulty Score to empower United Airlines' frontline teams with a proactive tool for resource planning, moving beyond reactive, experience-based assessments.

Our solution is a Hybrid Model that combines the best of both worlds:

- An Interpretable Rank-Based Scoring System that is transparent, outlier-proof, and easy for operational teams to understand and trust.

- A robust Machine Learning Validation Layer (Random Forest & Gradient Boosting) that confirms the accuracy of our approach, achieving 94.9% cross-validation accuracy.

Our approach uses a robust, **daily rank-based scoring engine**. This method is objective (no subjective weights), **outlier-proof**, and **operationally relevant**, as it adapts to the unique operational tempo of each day. The model analyzes over a dozen engineered features across multiple dimensions—including ground pressure, passenger complexity, baggage handling, and flight characteristics—to classify each flight as 'Easy', 'Medium', or 'Difficult'.

The final output is not just a theoretical model, but a production-ready pipeline, complete with validated performance, a justified $33M annual business case, and deployable SQL queries for immediate integration.


## 1.5. Key Features & Differentiators
What makes our solution ready to win and ready to deploy:

- Three-Layer Validation: Our model's effectiveness is proven by a 29x delay separation between 'Easy' and 'Difficult' flights, 92% accuracy on a temporal holdout week, and 94.9% cross-validation accuracy.

- Production-Ready SQL: We provide 7 tested SQL queries that replicate the entire data pipeline, ready for deployment in a production database.

- Justified Business Impact: A transparent, conservative ROI calculation projects over $33M in annual savings at ORD alone, based on industry-standard delay costs.

- Intelligent Baselines: We prove our model's necessity by showing it significantly outperforms simpler, common-sense baselines (like using only load factor or historical delay).

---

## 2. File Structure

The project is organized into the following structure:

```

    └── SkyHack_Submission_Innov8torX/
    │
    ├── 📁 Data/
    │   └── (All 5 raw CSV files for input)
    │
    ├── 📜 Code/
    │   ├── enhanced_main.py          # <-- PRIMARY SCRIPT: Run this to execute everything
    │   ├── flight_analyzer.py        #     (Contains base rank-based scoring logic)
    │   ├── enhanced_ml_analyzer.py   #     (Contains ML models, validation, and ROI)
    │   ├── additional_analysis.py    #     (Generates the 3 main presentation charts)
    │   ├── appendix_generator.py     #     (Generates all 8 appendix charts)
    │   ├── sql_queries.sql           #     (7 production-ready SQL queries for deployment)
    │   ├── test_all_sql.py           #     (Optional: Script to test the SQL queries)
    │   │
    │   ├── 📁 Charts/                 #     (All generated PNG charts are saved here)
    │   └── 📁 TXT/                    #     (All generated text reports are saved here)
    │
    │
    ├── 🚚 Deliverables/                          # Most Important contains all Results
    │   ├── Innov8torX_mini.pptx
    │   ├── test_Innov8torX_mini.csv and excel     # <-- PRIMARY DELIVERABLE: Final scored data
    │   ├── ml_validation_results.csv              #          (ML model validation results)
    │   ├── primary_drivers_chart.png              #     (Chart #1: Primary Drivers of Difficulty)
    │   ├── top_destinations_chart.png             #     (Chart #2: Most Difficult Destinations)
    │   ├── delay_validation_chart.png             #     (Chart #3: Delay Validation)
    │   │
    │   └── appendix_summary_stats.txt             #     Deliverable 1 Table
    │   └── business_impact_summary.txt
    │            
    └── 📖 README.md                   # (This file)

```


---

## 3. How to Run: Instructions for All Users
###    Requirements

To run this project, you will need the following installed:

* **Python:** Version 3.7 or higher
* **Python Libraries:**
    * `pandas`
    * `numpy`
    * `matplotlib`
    * `seaborn`

These libraries can be installed using `pip` as described in the setup instructions below.

It is highly recommended to use a **Python virtual environment** to manage dependencies and avoid conflicts. The steps are slightly different for Windows and Linux/macOS.

### Step A: Setup the Environment

#### For Linux and macOS Users:

1.  **Open our terminal** in the project's root directory.
2.  **Create a Virtual Environment**:
    ```bash
    python3 -m venv venv
    ```
3.  **Activate the Environment**:
    ```bash
    source venv/bin/activate
    cd Code/
    ```
    *(our terminal prompt should now start with `(venv)`)*.

#### For Windows Users:

1.  **Open Command Prompt or PowerShell** in the project's root directory.
2.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    ```
3.  **Activate the Environment**:
    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```
    ```powershell
    script .\venv\Scripts\activate
    cd .\Code\
    ```
    *(Your terminal prompt should now start with `(venv)`)*.

### Step B: Install Required Libraries

* With the virtual environment active, install all necessary packages using pip. This command works for all operating systems:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn pandasql
    ```

### Step C: Run the Analysis

The analysis is a two-step process. First, run the main scoring script, then run the visualization script. These commands are the same for all operating systems.

1.  **Run the Main Analysis**:
    * This performs all calculations and generates your final CSV files.
        ```bash
        python main.py
        ```
        for Generating Graphs and txt files run
        ```bash
          # Generate the 3 core presentation charts
          python additional_analysis.py

          # Generate the 8 detailed appendix charts
          python appendix_generator.py
        ```
    * Output:- 
    After running the commands, all generated files will be located inside the Code/ directory as follows:

      - Code/test_Innov8torX_mini.csv: The final, clean data deliverable for submission.

      -  Code/final_flight_data.csv: The rich helper file used for creating charts.

      -  Code/ml_validation_results.csv: A detailed report of the ML model performance.

      -  Code/TXT/business_impact_summary.txt: The full ROI calculation and assumptions.

      -   Code/TXT/appendix_summary_stats.txt: Summary statistics from the EDA.

      -  Code/Charts/*.png: All 11 PNG chart files, ready to be used in the presentation

       

---



