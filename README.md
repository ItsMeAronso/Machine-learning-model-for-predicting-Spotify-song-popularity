# Spotify Song Popularity Prediction

An end-to-end ML project that predicts whether a Spotify track will be popular (top 30% by popularity score), based on audio features, release year, genre and artist metadata. Framed as a business problem: which tracks deserve a limited marketing budget?

## Business Objective

For a record label, a wasted marketing budget (False Positive) is more costly than a missed hit (False Negative). Missing a hit is acceptable as long as the tracks we do invest in are real hits. The model is therefore tuned for **Precision** on the tracks it recommends, rather than overall accuracy or recall.

## Results

Evaluated on a held-out test set of 3,000 tracks (29.7% popular):

| Model | Precision | Recall | Tracks recommended |
|---|---|---|---|
| Random baseline (base rate) | 29.7% | - | - |
| Decision Tree (depth 4, class_weight='balanced') | 42.8% | 72.8% | 1,513 / 3,000 |
| **Random Forest (300 trees, threshold 0.5)** | **58.4%** | 21.0% | 320 / 3,000 |

The Random Forest recommends far fewer tracks, but about 58% of them are real hits, nearly double the base rate. The cost is low recall (it misses most hits), a trade-off chosen deliberately for the business goal above. The decision threshold was selected using 5-fold cross-validation on the training set only, then evaluated once on the test set.

The confusion matrices, feature importances and decision tree visualization are in the notebook.

## Key Insights

- **Artist popularity dominates.** `avg_artist_popularity` accounts for roughly 88% of the Decision Tree's feature importance, so the model mostly captures "popular artists make popular songs". Audio features (liveness, danceability, year, acousticness) add a smaller contribution.

## Data Pipeline

1. **Sampling:** 15,000 tracks sampled from the full dataset.
2. **Train/Test split (80/20)** performed before any preprocessing or statistics.
3. **Target definition:** binary `is_popular`, using the 70th percentile of popularity computed on the training set only.
4. **Feature engineering:** One-Hot Encoding for `genre`; top 15 niche sub-genres (selected on train only) parsed from nested string lists and encoded.
5. **Preprocessing:** `SimpleImputer` (mean) and `StandardScaler`, both fitted on train only.
6. **Models:** Decision Tree baseline (entropy, `class_weight='balanced'`) and Random Forest (300 trees, `min_samples_leaf=20`).
7. **Threshold tuning:** probability threshold chosen via `cross_val_predict` (5-fold) on the training set.

## Limitations

- Artist popularity dominates, so audio features alone may have limited predictive power (not yet tested without it).
- Recall is low by design; many potential hits are not recommended.
- Results come from a single train/test split, with no repeated runs or confidence intervals.
- A 15,000-track sample was used.

## Future Work

- Train a secondary model without artist-level metrics to isolate audio-driven prediction.
- Gradient boosting (XGBoost/LightGBM) and hyperparameter tuning.
- Temporal features (audio trends by decade or season).
- NLP on lyrics (sentiment, themes, explicit content flags).
