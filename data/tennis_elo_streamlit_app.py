import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression # Changed from LinearRegression
from sklearn.ensemble import RandomForestClassifier # Added RandomForest
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score # Added more metrics
from sklearn.preprocessing import OneHotEncoder # Added for surface encoding

# --- Page configuration and theme
st.set_page_config(
    page_title="Tennis Elo Match Predictor",
    layout="wide"
)
# Keep existing markdown styling

# --- Step 1: Load and preprocess data
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), 'data') # More robust path
    # Check if running in Streamlit Cloud or similar environment where 'data' might be top-level
    if not os.path.exists(data_dir):
         data_dir = "./data"

    years = [2022, 2023, 2024]
    dfs = []
    files_found = False
    for year in years:
        path = os.path.join(data_dir, f"atp_matches_{year}.csv")
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                dfs.append(df)
                files_found = True
            except Exception as e:
                st.error(f"Error loading {path}: {e}")
        else:
            st.warning(f"Data file not found: {path}")

    if not files_found:
        st.error("No data files found. Please ensure atp_matches_YYYY.csv files are in the 'data' directory.")
        st.stop() # Stop execution if no data is loaded

    data = pd.concat(dfs, ignore_index=True)
    # Select necessary columns early
    cols_to_keep = ['tourney_date', 'surface', 'winner_name', 'loser_name', 'best_of', 'winner_rank', 'loser_rank']
    data = data[cols_to_keep]

    # Basic cleaning
    data['tourney_date'] = pd.to_datetime(data['tourney_date'], format='%Y%m%d')
    data['surface'] = data['surface'].fillna('Hard').astype(str) # Ensure string type
    data = data.dropna(subset=['winner_name', 'loser_name']) # Drop rows with missing player names
    data = data.sort_values(by='tourney_date').reset_index(drop=True)
    return data

# --- Step 2: Calculate Elo ratings
@st.cache_data
def calculate_elo(data, k_factor=32, initial_elo=1500):
    elo_ratings = {}
    rank_ratings = {} # Store latest rank
    matches = []

    all_players = set(data['winner_name']).union(set(data['loser_name']))
    for player in all_players:
        elo_ratings[player] = initial_elo
        rank_ratings[player] = None # Initialize ranks

    for index, row in data.iterrows():
        winner, loser = row['winner_name'], row['loser_name']
        winner_rank, loser_rank = row['winner_rank'], row['loser_rank']

        # Ensure players are initialized
        if winner not in elo_ratings: elo_ratings[winner] = initial_elo
        if loser not in elo_ratings: elo_ratings[loser] = initial_elo

        elo_winner = elo_ratings[winner]
        elo_loser = elo_ratings[loser]

        # Store pre-match ratings and ranks
        current_match = {
            'tourney_date': row['tourney_date'],
            'surface': row['surface'],
            'winner_name': winner,
            'loser_name': loser,
            'winner_elo_before': elo_winner,
            'loser_elo_before': elo_loser,
            'winner_rank_before': winner_rank, # Use rank from current row
            'loser_rank_before': loser_rank,
            'best_of': row['best_of'],
            'outcome': 1 # 1 if player 1 (winner) wins, 0 otherwise (always 1 here as winner is fixed)
        }
        matches.append(current_match)

        # Update latest ranks
        if not pd.isna(winner_rank): rank_ratings[winner] = winner_rank
        if not pd.isna(loser_rank): rank_ratings[loser] = loser_rank


        # Elo calculation
        expected_winner = 1 / (1 + 10**((elo_loser - elo_winner) / 400))
        expected_loser = 1 / (1 + 10**((elo_winner - elo_loser) / 400))

        new_elo_winner = elo_winner + k_factor * (1 - expected_winner)
        new_elo_loser = elo_loser + k_factor * (0 - expected_loser)

        elo_ratings[winner] = new_elo_winner
        elo_ratings[loser] = new_elo_loser

    elo_df = pd.DataFrame(matches)
    # Convert latest ranks to a DataFrame for merging later if needed
    ranks_df = pd.DataFrame(list(rank_ratings.items()), columns=['player', 'current_rank'])
    return elo_df, elo_ratings, rank_ratings # Return rank_ratings dict too


# --- Step 3: Create Features (Modified)
@st.cache_data
def create_features(_elo_df): # Use _elo_df convention for cached functions
    df = _elo_df.copy()
    df['elo_diff'] = df['winner_elo_before'] - df['loser_elo_before']

    # Handle NaN ranks more robustly - use a very high rank like 2000
    df['winner_rank_filled'] = df['winner_rank_before'].fillna(2000)
    df['loser_rank_filled'] = df['loser_rank_before'].fillna(2000)
    df['rank_diff'] = df['winner_rank_filled'] - df['loser_rank_filled']

    # -- Feature Engineering for Surface --
    # Use OneHotEncoder for surface - this is more standard for ML
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    surface_encoded = encoder.fit_transform(df[['surface']])
    # Create meaningful column names based on categories
    surface_feature_names = [f'surface_{cat}' for cat in encoder.categories_[0]]
    surface_df = pd.DataFrame(surface_encoded, columns=surface_feature_names, index=df.index)

    # Combine original df with new features
    df = pd.concat([df, surface_df], axis=1)

    # --- Define features for the model ---
    # Base features + surface features
    feature_cols = ['elo_diff', 'rank_diff', 'best_of'] + surface_feature_names
    # Ensure only necessary columns are present for X
    X = df[feature_cols]
    y = df['outcome'] # Outcome is always 1 in this setup, we need to reframe

    # --- Reframe for Prediction Task ---
    # We need examples of both wins and losses for the *same* match features
    # Create mirrored entries where player roles are swapped
    df_swapped = df.copy()
    df_swapped.rename(columns={
        'winner_name': 'p2_name', 'loser_name': 'p1_name',
        'winner_elo_before': 'p2_elo', 'loser_elo_before': 'p1_elo',
        'winner_rank_filled': 'p2_rank', 'loser_rank_filled': 'p1_rank'
    }, inplace=True)
    df.rename(columns={
        'winner_name': 'p1_name', 'loser_name': 'p2_name',
        'winner_elo_before': 'p1_elo', 'loser_elo_before': 'p2_elo',
        'winner_rank_filled': 'p1_rank', 'loser_rank_filled': 'p2_rank'
    }, inplace=True)

    df['outcome'] = 1 # Player 1 (original winner) won
    df_swapped['outcome'] = 0 # Player 1 (original loser) lost

    # Combine original and swapped DataFrames
    full_df = pd.concat([df, df_swapped], ignore_index=True)

    # Recalculate diffs based on Player 1 vs Player 2 perspective
    full_df['elo_diff'] = full_df['p1_elo'] - full_df['p2_elo']
    full_df['rank_diff'] = full_df['p1_rank'] - full_df['p2_rank'] # Already handles NaNs

    # Final Feature Set + Target
    X = full_df[feature_cols]
    y = full_df['outcome']

    return X, y, feature_cols, encoder # Return encoder for prediction use

# --- Step 4: Train Models (Modified)
#@st.cache_data # Cannot cache models directly easily in Streamlit
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {}
    accuracies = {}
    log_losses = {}
    aucs = {}

    # Decision Tree
    dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced') # Added balanced weights
    dt_clf.fit(X_train, y_train)
    y_pred_dt = dt_clf.predict(X_test)
    y_prob_dt = dt_clf.predict_proba(X_test)[:, 1]
    models['Decision Tree'] = dt_clf
    accuracies['Decision Tree'] = accuracy_score(y_test, y_pred_dt)
    log_losses['Decision Tree'] = log_loss(y_test, y_prob_dt)
    aucs['Decision Tree'] = roc_auc_score(y_test, y_prob_dt)
    dt_feat_imp = dt_clf.feature_importances_

    # Logistic Regression
    log_reg = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000) # Added balanced weights
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    y_prob_lr = log_reg.predict_proba(X_test)[:, 1]
    models['Logistic Regression'] = log_reg
    accuracies['Logistic Regression'] = accuracy_score(y_test, y_pred_lr)
    log_losses['Logistic Regression'] = log_loss(y_test, y_prob_lr)
    aucs['Logistic Regression'] = roc_auc_score(y_test, y_prob_lr)

    # Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1) # Added balanced weights
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    y_prob_rf = rf_clf.predict_proba(X_test)[:, 1]
    models['Random Forest'] = rf_clf
    accuracies['Random Forest'] = accuracy_score(y_test, y_pred_rf)
    log_losses['Random Forest'] = log_loss(y_test, y_prob_rf)
    aucs['Random Forest'] = roc_auc_score(y_test, y_prob_rf)


    return models, accuracies, log_losses, aucs, dt_feat_imp

# --- Main execution ---
data = load_data()

if not data.empty:
    elo_df, elo_ratings_final, rank_ratings_final = calculate_elo(data)
    X, y, feature_cols, surface_encoder = create_features(elo_df) # Get encoder

    # Train models
    models, accuracies, log_losses, aucs, dt_feat_imp = train_models(X, y)

    # Get list of players with calculated Elo
    player_list = sorted(list(elo_ratings_final.keys()))

    # --- UI Layout ---
    st.title("🎾 Tennis Elo Match Predictor")

    # Model metrics
    st.header("Model Performance")
    st.write("Models trained on ATP matches 2022-2024, predicting Player 1 win.")
    perf_data = {
        "Model": list(models.keys()),
        "Accuracy": [f"{acc:.2%}" for acc in accuracies.values()],
        "Log Loss": [f"{ll:.4f}" for ll in log_losses.values()],
        "AUC": [f"{auc:.4f}" for auc in aucs.values()]
     }
    st.dataframe(pd.DataFrame(perf_data))
    st.caption("Accuracy: Proportion of correct predictions. Log Loss: Lower is better, measures probability accuracy. AUC: Higher is better, measures model's ability to distinguish classes.")


    # Simulation
    st.header("Simulate a Match Between Two Players")

    model_choice = st.selectbox("Select Model for Prediction:", list(models.keys()))
    selected_clf = models[model_choice]

    col1, col2 = st.columns(2)

    # Get unique surfaces from the encoder
    surface_options = list(surface_encoder.categories_[0])

    with col1:
        player_a = st.selectbox("Player A", player_list, index=player_list.index("Carlos Alcaraz") if "Carlos Alcaraz" in player_list else 0)
        player_b = st.selectbox("Player B", player_list, index=player_list.index("Novak Djokovic") if "Novak Djokovic" in player_list else 1)
    with col2:
        surface_input = st.selectbox("Surface", surface_options)
        best_of_input = st.selectbox("Match Type (sets)", [3,5])

    # Display Elo & Rank
    elo_a = elo_ratings_final.get(player_a, 1500) # Use get with default
    elo_b = elo_ratings_final.get(player_b, 1500)
    rank_a = rank_ratings_final.get(player_a) # Use get
    rank_b = rank_ratings_final.get(player_b)
    st.markdown(f"<h4>{player_a}: Elo {elo_a:.0f}, Rank {rank_a or 'N/A'}</h4>", unsafe_allow_html=True)
    st.markdown(f"<h4>{player_b}: Elo {elo_b:.0f}, Rank {rank_b or 'N/A'}</h4>", unsafe_allow_html=True)


    # Build feature row for prediction
    if player_a == player_b:
        st.warning("Please select two different players.")
    else:
        # Calculate differences from Player A's perspective
        fd = elo_a - elo_b
        # Use filled rank NAs for diff calculation
        rank_a_filled = rank_a if rank_a is not None else 2000
        rank_b_filled = rank_b if rank_b is not None else 2000
        rd = rank_a_filled - rank_b_filled

        # Prepare one-hot encoded surface features for the selected surface
        surface_encoded_input = surface_encoder.transform([[surface_input]])
        surface_feature_dict = dict(zip(surface_encoder.get_feature_names_out(['surface']), surface_encoded_input[0]))

        # Create the full feature dictionary
        feature_dict = {
            'elo_diff': fd,
            'rank_diff': rd,
            'best_of': best_of_input,
            **surface_feature_dict # Add surface features
        }

        # Ensure columns are in the correct order as used in training
        # Create DataFrame with all feature columns, initialized to 0, then update
        predict_data = {col: [0] for col in feature_cols} # Initialize all features to 0
        for key, value in feature_dict.items():
             if key in predict_data: # Only update if feature was used in training
                 predict_data[key] = [value]

        features_df = pd.DataFrame(predict_data)[feature_cols] # Ensure column order


        # Predict
        try:
            prob_a_wins = selected_clf.predict_proba(features_df)[0][1] # Probability of class 1 (Player A wins)
            pred_winner = player_a if prob_a_wins > 0.5 else player_b
            prob_display = prob_a_wins if pred_winner == player_a else 1 - prob_a_wins # Probability of the predicted winner

            st.markdown(f"<h3>Predicted Winner ({model_choice}): {pred_winner}</h3>", unsafe_allow_html=True)
            st.markdown(f"<h4>Win Probability: {prob_display:.1%}</h4>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.error(f"Feature Dict: {feature_dict}")
            st.error(f"Features DF shape: {features_df.shape}")
            st.error(f"Features DF columns: {features_df.columns.tolist()}")
            st.error(f"Model expected columns: {feature_cols}")


    # Feature Importance at end (from Decision Tree)
    st.header("Decision Tree Feature Importance")
    st.caption("Importance values from the Decision Tree model, showing relative influence on predictions.")
    try:
        # Filter out features with 0 importance for cleaner plot
        imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': dt_feat_imp})
        imp_df = imp_df[imp_df['Importance'] > 0.001].sort_values(by='Importance', ascending=False)

        if not imp_df.empty:
           fig, ax = plt.subplots(figsize=(8, 4)) # Adjusted size
           sns.barplot(x='Importance', y='Feature', data=imp_df, ax=ax, palette='viridis') # Changed orientation
           ax.set_xlabel("Importance")
           ax.set_ylabel("Feature")
           st.pyplot(fig)
        else:
            st.write("No features had importance > 0.001 according to the Decision Tree.")

    except Exception as e:
        st.error(f"Could not plot feature importance: {e}")
else:
    st.error("Data could not be loaded. Cannot proceed.")




