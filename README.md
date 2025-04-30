# ATP Tennis Match Predictor with Elo Ratings

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python) ![Pandas](https://img.shields.io/badge/Pandas-Used-blue?logo=pandas) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Used-orange?logo=scikit-learn) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)

## Overview

This project implements an interactive web application using Streamlit to predict the outcome of ATP tennis matches. It leverages historical match data (2022-2024) to calculate player Elo ratings and trains machine learning models (Decision Tree, Logistic Regression, Random Forest) to predict win probability based on Elo difference, rank difference, court surface, and match format (best-of sets).

The primary goal was to build an end-to-end prediction tool, from data processing and feature engineering to model training and interactive prediction via a user-friendly web interface.

ADDED LINK for instant deployment
`[Live Demo Link](https://larsdukart-tennis-elo-app-datatennis-elo-streamlit-app-xk6ntk.streamlit.app)`

## Features

* Loads and preprocesses ATP match data from 2022-2024.
* Calculates player Elo ratings iteratively based on match outcomes.
* Engineers features including Elo difference, rank difference, and **one-hot encoded court surface**.
* Trains and evaluates three different classification models:
    * Decision Tree
    * Logistic Regression
    * Random Forest
* Provides performance metrics (Accuracy, Log Loss, AUC) for model comparison.
* Offers an interactive Streamlit interface to:
    * Select two players.
    * Choose the court surface and match format (Best of 3 / Best of 5).
    * Select a model for prediction.
    * View predicted winner and win probability.
    * Display Decision Tree feature importances.

## Data

The model is trained using publicly available ATP match results data from 2022, 2023, and 2024, including details like players, date, surface, best-of sets, and player ranks.

*(Note: Ensure data source compliance if sharing publicly).*

## Methodology

1.  **Data Processing:** Match data from CSV files is loaded, cleaned (handling missing values, date conversion), and sorted chronologically.
2.  **Elo Calculation:** A standard Elo rating system is implemented. Ratings are initialized (1500) and updated after each match based on the outcome and the K-factor (set to 32).
3.  **Feature Engineering:** For model training, the following features are created for each match from the perspective of 'Player 1':
    * `elo_diff`: Player 1 Elo - Player 2 Elo (before the match).
    * `rank_diff`: Player 1 Rank - Player 2 Rank (missing ranks imputed with 2000).
    * `best_of`: Integer representing match format (3 or 5).
    * `surface_*`: **One-hot encoded features representing the court surface** (e.g., `surface_Hard`, `surface_Clay`, `surface_Grass`).
    * To create a balanced training set, each match generates two rows: one for the actual winner as Player 1 (outcome=1) and one swapping the players (outcome=0).
4.  **Modeling:** The engineered features are used to train three Scikit-learn classifiers. Performance is evaluated on a hold-out test set (20% of the data) using Accuracy, Log Loss, and Area Under the Curve (AUC).

## Project Iteration: Incorporating Court Surface

A key challenge during development was ensuring the model appropriately considered the court surface, a known critical factor in tennis. Initially, the surface feature was inadvertently excluded from the training data. This was rectified by:

1.  Implementing one-hot encoding for the `surface` category.
2.  **Explicitly including these `surface_*` features in the feature set (`X`) used for model training.**
3.  Ensuring the prediction pipeline correctly encoded and used the surface input selected by the user.

This iterative process significantly improved the model's ability to factor in surface characteristics when making predictions, demonstrating a practical problem-solving approach.

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/larsdukart/tennis-elo-app](https://github.com/larsdukart/tennis-elo-app)
    cd tennis-elo-app
    ```
2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Ensure data files are present:** The `data/` folder containing `atp_matches_*.csv` files should be in the main directory.
5.  **Run the Streamlit app (make sure you are inside the `tennis-elo-app` directory):**
    ```bash
    streamlit run tennis_elo_streamlit_app.py
    ```

## Current Status & Future Work

This project provides a functional baseline for tennis match prediction using Elo ratings and standard ML models within an interactive app.

**Limitations & Potential Improvements:**

* **Standard Elo:** The current Elo system doesn't account for surface-specific player strengths. Implementing surface-specific Elo ratings would likely improve accuracy significantly.
* **Basic Hyperparameter Tuning:** Models use default or minimally tuned hyperparameters. Implementing systematic tuning (e.g., `GridSearchCV` or `RandomizedSearchCV`) could optimize performance.
* **Evaluation:** Relies on a single train-test split. Using cross-validation would provide more robust performance estimates.
* **Feature Set:** More features could be engineered (e.g., player age, recent form metrics, head-to-head stats).
* **Code Structure:** The main logic is in a single script; refactoring into modules could improve maintainability.
* **Deployment:** The app is currently set up for local execution. Deploying it (e.g., via Streamlit Community Cloud) would make it publicly accessible.

This project demonstrates core data science skills including data wrangling, feature engineering, model building, evaluation, and basic web application development. The identified future work outlines clear paths for further enhancing its predictive power and robustness.