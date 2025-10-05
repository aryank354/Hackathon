# SkyHack Challenge: Flight Difficulty Score

## 1. Project Overview

This project addresses the SkyHack Challenge by developing a systematic, data-driven **Flight Difficulty Score**. The goal is to move beyond inconsistent, experience-based assessments and provide United Airlines frontline teams with a proactive tool for resource planning.

Our approach uses a robust, **daily rank-based scoring engine**. This method is objective (no subjective weights), **outlier-proof**, and **operationally relevant**, as it adapts to the unique operational tempo of each day. The model analyzes over a dozen engineered features across multiple dimensions—including ground pressure, passenger complexity, baggage handling, and flight characteristics—to classify each flight as 'Easy', 'Medium', or 'Difficult'.

---

## 2. File Structure

The project is organized into the following structure:

```
└── SkyHack_Submission_Innov8torX/
    │
    ├── 📁 Data/
    │   ├── Flight_Level_Data.csv
    │   ├── PNR_Flight_Level_Data.csv
    │   ├── PNR_Remark_Level_Data.csv
    │   ├── Bag_Level_Data.csv
    │   └── Airports_Data.csv
    │
    ├── 📜 Code/
    │   ├── main.py                 # <-- Driver script to run the full analysis
    │   ├── flight_analyzer.py      # <-- Core logic, feature engineering, and scoring
    │   └── additional_analysis.py  # <-- Generates all charts
    │
    ├── 🚚 Deliverables/
    │   ├── SkyHack Challenge (Innov8torX).pptx  # <-- FINAL PRESENTATION
    │   ├── test_Innov8torX_mini.csv             # <-- FINAL SUBMISSION CSV
    │   └── test_Innov8torX_mini.xlsx            # <-- (Optional) FINAL SUBMISSION EXCEL
    │
    ├── 📊 Generated_Charts/
    │   ├── primary_drivers_chart.png       # <-- FINAL CHART 1
    │   ├── top_destinations_chart.png      # <-- FINAL CHART 2
    │   └── delay_validation_chart.png      # <-- FINAL CHART 3
    │
    │
    └── 📖 README.md                         # <-- Instructions to run the project

```

### File Descriptions


* **`Data/`**: This folder should contain the five raw CSV datasets provided for the challenge.
* **`flight_analyzer.py`**: The core engine of the project. Contains the `RankBasedFlightDifficultyScorer` class for data processing and scoring.
* **`main.py`**: The primary execution script. It runs the entire analysis pipeline from start to finish. It produces two key files:
    1.  `test_Innov8torX_mini.csv` (our final deliverable).


* **`test_Innov8torX_mini.csv`**: **(Generated Output)** The primary data deliverable. This CSV file contains the final difficulty score and classification for each flight.    
* **`additional_analysis.py`**: This script **consumes the `final_flight_data.csv` file** to generate the three key charts required for the presentation.
* **`README.md`**: This instruction file.

* **`*.png`**: **(Generated Output)** The three chart images created by `additional_analysis.py`, used in the final presentation.




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
    ```
    *(Your terminal prompt should now start with `(venv)`)*.

### Step B: Install Required Libraries

* With the virtual environment active, install all necessary packages using pip. This command works for all operating systems:
    ```bash
    pip install pandas numpy matplotlib seaborn
    ```

### Step C: Run the Analysis

The analysis is a two-step process. First, run the main scoring script, then run the visualization script. These commands are the same for all operating systems.

1.  **Run the Main Analysis**:
    * This performs all calculations and generates your final CSV files.
        ```bash
        python main.py
        ```
    * Output:- 
        1.  `test_Innov8torX_mini.csv` (our final deliverable).

2.  **Generate Presentation Visuals**:
    * After the main analysis is complete, run this script to create our charts.
        ```bash
        python additional_analysis.py
        ```
    * Output: Three PNG files:
        1.  `primary_drivers_chart.png`
        2.  `top_destinations_chart.png`
        3.  `delay_validation_chart.png`        

---

## 4. Expected Output

After running both scripts, our project folder will contain:

* **`test_Innov8torX_mini.csv`**: our final, clean data deliverable for submission.
* **`primary_drivers_chart.png`**: Chart showing the #1 causes of flight difficulty.
* **`top_destinations_chart.png`**: Chart showing the most challenging destinations.
* **`delay_validation_chart.png`**: Chart proving the model's correlation with real-world delays.

