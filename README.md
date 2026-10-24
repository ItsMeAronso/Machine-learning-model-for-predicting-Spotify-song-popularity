# Spotify Song Popularity Prediction & Analysis

An end-to-end Machine Learning project aimed at predicting whether a song on Spotify will achieve high popularity based on its audio features, release year, and artist metadata.

## 📌 Project Overview & Objective
In the music industry, understanding what drives a song's success is crucial. This project builds a **Decision Tree Classifier** to classify tracks as "Popular" (top 30% most popular tracks based on a 70th percentile threshold) or "Not Popular". 

Special care was taken to address potential biases (such as the overwhelming dominance of an artist's prior fame) to evaluate the true impact of musical and acoustic characteristics.

---

## 🔄 Data Pipeline & Workflow
To ensure rigorous methodology and prevent **Data Leakage**, the following pipeline was implemented:
1. **Stratified Train/Test Split (80/20):** The dataset was split prior to any preprocessing or statistical calculations.
2. **Target Definition:** A binary target variable (`is_popular`) was established by calculating the 70th percentile threshold exclusively on the **training set**.
3. **Feature Engineering & Encoding:**
   - **One-Hot Encoding** was applied to the main `genre` feature.
   - **Niche Genres Parsing:** Extracted and encoded the top 15 most frequent sub-genres from nested string lists.
4. **Imputation & Scaling:** 
   - Handled missing values using `SimpleImputer` (fitted on train).
   - Normalized numerical features using `StandardScaler` (fitted on train).
5. **Model Training:** Trained a `DecisionTreeClassifier` utilizing entropy and class balancing (`class_weight='balanced'`) to handle class distribution.

---

## 📊 Key Results & Insights
* **The Dominance of Artist Popularity:** Initial feature importance and decision tree splitting indicate that an artist's historical popularity (`avg_artist_popularity`) acts as a primary gatekeeper for track success.
* **The Power of Audio Features:** Once past the artist popularity threshold, the model successfully leverages secondary audio and contextual features—such as `danceability`, release `year`, and specific `genre` classifications—to refine its predictions.
* **Model Performance:** The classification report and confusion matrix demonstrate a balanced trade-off in identifying popular tracks, proving that the model looks beyond mere brand recognition.


---

## 🌳 Decision Tree Visualization
A snippet of the top 3 levels of the trained Decision Tree, illustrating how the model splits data based on artist metrics and track attributes:

---

## 💡 Hypotheses & Recommendations for Future Work
To further improve predictive power and reduce potential dataset limitations, future iterations of this project should explore:
1. **Advanced Temporal Features:** Incorporating audio trends over specific decades or seasons to see if listener preferences for specific acousticness or energy shift over time.
2. **Natural Language Processing (NLP):** Analyzing the `lyrics` column (e.g., sentiment analysis, word embeddings, or explicit content flags) to test whether lyrical themes correlate with popularity.
3. **Ensemble Models:** Upgrading from a single Decision Tree to Random Forests or Gradient Boosting (XGBoost/LightGBM) to capture non-linear interactions and boost accuracy metrics.
4. **Handling Artist Bias:** Training a secondary model *without* artist-level metrics to isolate pure audio-driven hit prediction.

5.**"From a business perspective:** for a record label, Precision is prioritized over Recall. While missing a potential hit (False Negative) represents a missed opportunity, allocating a limited marketing budget to a failing track (False Positive) results in a direct financial loss. Therefore, tuning the decision threshold to maximize Precision ensures high confidence in marketing investments
