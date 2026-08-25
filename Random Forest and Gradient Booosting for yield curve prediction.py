import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from google.colab import files
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.base import clone
from scipy.stats import t as student_t
import warnings
warnings.filterwarnings('ignore')

uploaded = files.upload()

file_path = "datasetML.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')
date_col = df.columns[0]
df[date_col] = pd.to_datetime(df[date_col], format='%d.%m.%Y', dayfirst=True)
df.set_index(date_col, inplace=True)
maturity_cols = df.columns
df = df.apply(lambda x: x.astype(str).str.replace(',', '.').astype(float))
df.sort_index(inplace=True)

# Random Forest model

last_known_date = df.index[-1]
last_known_yields = df.iloc[-1].values

def create_features_target(data, target_horizon=1):

    X_list = []
    y_list = []

    dates = data.index

    for i in range(len(data) - target_horizon):

        current_yields = data.iloc[i].values
        current_date = dates[i]
        year = current_date.year
        features = np.concatenate([current_yields, [year]])
        X_list.append(features)
        y_list.append(data.iloc[i + target_horizon].values)

    return np.array(X_list), np.array(y_list), dates

X, y, dates_used = create_features_target(df, target_horizon=1)

def multioutput_rmse(y_true, y_pred):
    mse = mean_squared_error(
        y_true,
        y_pred,
        multioutput="uniform_average"
    )
    return np.sqrt(mse)

rmse_scorer = make_scorer( multioutput_rmse, greater_is_better=False )

base_rf = RandomForestRegressor(
    criterion="squared_error",
    bootstrap=True,
    random_state=42,
    n_jobs=1
)

param_grid = {

    "n_estimators": [100, 200, 400],

    "max_depth": [5, 10, None],

    "min_samples_split": [2, 5],

    "min_samples_leaf": [1, 2],

    "max_features": ["sqrt", 1.0],

    "max_samples": [0.8, None]
}

tscv = TimeSeriesSplit(n_splits=5)

grid_search = GridSearchCV(
    estimator=base_rf,
    param_grid=param_grid,
    scoring=rmse_scorer,
    cv=tscv,
    refit=True,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid_search.fit(X, y)

print("\n" + "=" * 70)
print("BEST RANDOM FOREST PARAMETERS")
print("=" * 70)

print(grid_search.best_params_)

rf = grid_search.best_estimator_
rf.set_params(n_jobs=-1)
best_cv_rmse = -grid_search.best_score_
print(f"\nBest Average CV RMSE: {best_cv_rmse:.6f}")

grid_results = pd.DataFrame(grid_search.cv_results_)
grid_results["Mean_CV_RMSE"] = -grid_results["mean_test_score"]
grid_results["Std_CV_RMSE"] = grid_results["std_test_score"]

# Gradient Boosting model

base_xgb = xgb.XGBRegressor(

    objective='reg:squarederror',
    multi_strategy="multi_output_tree",
    random_state=42,
    n_jobs=1,
    tree_method='hist',
    device='cpu'
)

param_grid_xgb = {

    "n_estimators": [200, 300,400,500],
    "max_depth": [1, 2,3],
    "learning_rate": [0.03, 0.04, 0.05],
    "min_child_weight": [3, 5],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "reg_lambda": [0.1, 0.5],
    "reg_alpha": [ 0.0,0.01, 0.02]
}

print(f"\nXGBoost Parameters for Grid Search::")
num_combinations_xgb = np.prod([len(v) for v in param_grid_xgb.values()])
print(f"  Total number of combinations: {num_combinations_xgb}")
print(f"  Total number of models: {num_combinations_xgb * 5}")

grid_search_xgb = GridSearchCV(
    estimator=base_xgb,
    param_grid=param_grid_xgb,
    scoring=rmse_scorer,
    cv=tscv,
    refit=True,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid_search_xgb.fit(X, y)

xgb_best = grid_search_xgb.best_estimator_
xgb_best.set_params(n_jobs=-1)

best_cv_rmse_xgb = -grid_search_xgb.best_score_

print(f"\nBEST XGBOOST PARAMETERS:")
for param, value in grid_search_xgb.best_params_.items():
    print(f"  {param}: {value}")
print(f"  Best CV RMSE: {best_cv_rmse_xgb:.6f}")

grid_results_xgb = pd.DataFrame(grid_search_xgb.cv_results_)
grid_results_xgb["Mean_CV_RMSE"] = -grid_results_xgb["mean_test_score"]
grid_results_xgb["Std_CV_RMSE"] = grid_results_xgb["std_test_score"]

# Random Forest Feature Importance

feature_names = list(maturity_cols) + ["year"]

importances = rf.feature_importances_
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

print("\n" + "="*60)
print("FEATURE IMPORTANCE RANDOM FOREST")
print("="*60)
print(importance_df)

plt.figure(figsize=(10, 7))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.gca().invert_yaxis()
plt.xlabel("Relative importance")
plt.ylabel("Input variable")
plt.title("Random Forest Feature Importance")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

# Gradient Boosting Feature Importance.

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE GRADIENT BOOSTING")
print("=" * 80)

feature_names = list(maturity_cols) + ['year']
importances = xgb_best.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(importance_df.head(10).to_string(index=False))
plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Relative Importance")
plt.ylabel("Input variable")
plt.title("XGBoost Feature Importance")

# Najvažniji feature na vrhu
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ============================================================================
# RECURSIVE FORECASTING – XGBOOST
# ============================================================================

target_dates = pd.date_range(start='2023-07-01', end='2023-12-01', freq='MS')

current_yields = last_known_yields.copy()
current_date = pd.Timestamp('2023-06-01 00:00:00')
predictions_xgb = {}

for step, future_date in enumerate(target_dates, start=1):
    year = current_date.year
    features = np.concatenate([current_yields, [year]]).reshape(1, -1)

    next_yields = xgb_best.predict(features)[0]
    predictions_xgb[future_date] = next_yields

    current_yields = next_yields
    current_date = future_date

july_2023_xgb = predictions_xgb.get(pd.Timestamp('2023-07-01'))
dec_2023_xgb = predictions_xgb.get(pd.Timestamp('2023-12-01'))

print("\n" + "=" * 80)
print("1 - Step ahead July 2023 Gradient Boosting ")
print("=" * 80)

print(f"\n{'Ročnost':<15}  {'XGBoost':<20}")
print("-" * 70)

for i, col in enumerate(maturity_cols):
    xgb_val = july_2023_xgb[i]
    print(f"{col:<15} {xgb_val:>19.6f} ")


print("\n" + "=" * 80)
print("6 - Step ahead December 2023 Gradient Boosting")
print("=" * 80)

print(f"\n{'Ročnost':<15} {'XGBoost':<20} ")
print("-" * 70)

for i, col in enumerate(maturity_cols):
    xgb_val = dec_2023_xgb[i]
    print(f"{col:<15}  {xgb_val:>19.6f} ")


target_dates = pd.date_range(start='2023-07-01', end='2023-12-01', freq='MS')

current_yields = last_known_yields.copy()
current_date = pd.Timestamp('2023-06-01 00:00:00')
predictions = {}

for step, future_date in enumerate(target_dates, start=1):

    month = current_date.month
    year = current_date.year

    features = np.concatenate([current_yields, [year]]).reshape(1, -1)

    next_yields = rf.predict(features)[0]

    predictions[future_date] = next_yields

    current_yields = next_yields
    current_date = future_date


july_2023_pred = predictions.get(pd.Timestamp('2023-07-01'))
dec_2023_pred = predictions.get(pd.Timestamp('2023-12-01'))

# ----------------------------
# 5. Prikaz rezultata
# ----------------------------
print("\n" + "="*60)
print("Random forest")
print("="*60)

if july_2023_pred is not None:
    print("\n📅 July 2023.")
    for i, col in enumerate(maturity_cols):
        print(f"{july_2023_pred[i]:.6f}")

if dec_2023_pred is not None:
    print("\n📅 Decembar 2023.")
    for i, col in enumerate(maturity_cols):
        print(f" {dec_2023_pred[i]:.6f}")


import pandas as pd

# ----------------------------
# Gradient Boosting / XGBoost
# ----------------------------

gb_july_df = pd.DataFrame({
    "Ročnost": maturity_cols,
    "XGBoost": july_2023_xgb
})

gb_dec_df = pd.DataFrame({
    "Ročnost": maturity_cols,
    "XGBoost": dec_2023_xgb
})

# Čuvanje u Excel
with pd.ExcelWriter("gradient_boosting_predictions.xlsx", engine="openpyxl") as writer:
    gb_july_df.to_excel(
        writer,
        sheet_name="July_2023",
        index=False
    )

    gb_dec_df.to_excel(
        writer,
        sheet_name="December_2023",
        index=False
    )

print("Gradient Boosting results are saved in gradient_boosting_predictions.xlsx")


# ----------------------------
# Random Forest
# ----------------------------

rf_july_df = pd.DataFrame({
    "Ročnost": maturity_cols,
    "Random_Forest": july_2023_pred
})

rf_dec_df = pd.DataFrame({
    "Ročnost": maturity_cols,
    "Random_Forest": dec_2023_pred
})

# Čuvanje u Excel
with pd.ExcelWriter("random_forest_predictions.xlsx", engine="openpyxl") as writer:
    rf_july_df.to_excel(
        writer,
        sheet_name="July_2023",
        index=False
    )

    rf_dec_df.to_excel(
        writer,
        sheet_name="December_2023",
        index=False
    )

print("Random Forest results are saved in random_forest_predictions.xlsx")


# =============================================================================
# DIEBOLD-MARIANO TEST:
# RANDOM FOREST VS XGBOOST
# EXPANDING-WINDOW PSEUDO-OUT-OF-SAMPLE EVALUATION
# =============================================================================


maturity_cols = df.columns

df = df.apply(lambda x: x.astype(str).str.replace(',', '.').astype(float))

from sklearn.base import clone
from scipy.stats import t as student_t


rf_selected = globals().get("rf_best", globals().get("rf", None))

if rf_selected is None:
    raise NameError(
        "Random Forest model nije pronađen. "
        "Postavi rf_selected = rf ili rf_selected = rf_best."
    )

if "xgb_best" not in globals():
    raise NameError(
        "XGBoost model xgb_best nije pronađen."
    )


rf_template = clone(rf_selected)
xgb_template = clone(xgb_best)

rf_template.set_params(n_jobs=-1)
xgb_template.set_params(n_jobs=-1)


def recursive_forecast_path( model,initial_yields,origin_period, max_horizon ):

    current_yields = np.asarray(
        initial_yields,
        dtype=float
    ).copy()

    current_period = origin_period
    forecasts = []

    for step in range(1, max_horizon + 1):

        # Isti skup atributa kao u funkciji create_features_target:
        # 16 prinosa + godina tekućeg meseca
        features = np.concatenate(
            [current_yields, [current_period.year]]
        ).reshape(1, -1)

        next_yields = np.asarray(
            model.predict(features),
            dtype=float
        ).reshape(-1)

        if next_yields.size != len(initial_yields):
            raise ValueError(
                "Model nije vratio očekivani broj prinosa. "
                f"Očekivano: {len(initial_yields)}, "
                f"dobijeno: {next_yields.size}."
            )

        forecasts.append(next_yields.copy())

        # Prognozirana kriva postaje ulaz za sledeći korak
        current_yields = next_yields
        current_period = current_period + 1

    return forecasts


# -------------------------------------------------------------------------
# 3. EXPANDING-WINDOW BACKTEST
# -------------------------------------------------------------------------
def expanding_window_backtest(
    data,
    rf_model_template,
    xgb_model_template,
    horizons=(1, 6),
    evaluation_start="2018-01",
    verbose=True
):

    data = data.copy()
    data = data.sort_index()
    data = data.astype(float)

    if data.isna().any().any():
        raise ValueError(
            "Podaci sadrže nedostajuće vrednosti. "
            "Potrebno ih je obraditi pre backtest postupka."
        )

    horizons = tuple(sorted(set(horizons)))

    if any(h < 1 for h in horizons):
        raise ValueError("Svi prognozni horizonti moraju biti pozitivni.")

    max_horizon = max(horizons)

    periods = data.index.to_period("M")
    evaluation_start_period = pd.Period(
        evaluation_start,
        freq="M"
    )

    eligible_targets = np.flatnonzero(
        periods >= evaluation_start_period
    )

    if len(eligible_targets) == 0:
        raise ValueError(
            "Izabrani početak evaluacije nije prisutan u podacima."
        )

    first_target_index = int(eligible_targets[0])

    # Za h=6 prvi forecast origin mora biti šest meseci
    # pre prvog ciljnog datuma.
    first_origin_index = max(
        1,
        first_target_index - max_horizon
    )

    results = {
        h: {
            "dates": [],
            "actual": [],
            "rf": [],
            "xgb": []
        }
        for h in horizons
    }

    total_origins = len(data) - 1 - first_origin_index

    for counter, origin_index in enumerate(
        range(first_origin_index, len(data) - 1),
        start=1
    ):

        origin_date = data.index[origin_index]
        origin_period = periods[origin_index]

        # Trening skup sadrži samo podatke dostupne
        # zaključno sa forecast origin-om.
        training_data = data.iloc[:origin_index + 1]

        X_train, y_train, _ = create_features_target(
            training_data,
            target_horizon=1
        )

        if len(X_train) == 0:
            continue

        # Nova kopija modela za svaki forecast origin
        rf_model = clone(rf_model_template)
        xgb_model = clone(xgb_model_template)

        rf_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)

        available_horizon = min(
            max_horizon,
            len(data) - 1 - origin_index
        )

        rf_path = recursive_forecast_path(
            model=rf_model,
            initial_yields=data.iloc[origin_index].values,
            origin_period=origin_period,
            max_horizon=available_horizon
        )

        xgb_path = recursive_forecast_path(
            model=xgb_model,
            initial_yields=data.iloc[origin_index].values,
            origin_period=origin_period,
            max_horizon=available_horizon
        )

        for horizon in horizons:

            if horizon > available_horizon:
                continue

            target_index = origin_index + horizon
            target_period = periods[target_index]

            if target_period < evaluation_start_period:
                continue

            results[horizon]["dates"].append(
                data.index[target_index]
            )

            results[horizon]["actual"].append(
                data.iloc[target_index].values.astype(float)
            )

            results[horizon]["rf"].append(
                rf_path[horizon - 1]
            )

            results[horizon]["xgb"].append(
                xgb_path[horizon - 1]
            )

        if verbose and (
            counter == 1
            or counter % 12 == 0
            or counter == total_origins
        ):
            print(
                f"Obrađen forecast origin: "
                f"{origin_date.strftime('%Y-%m')} "
                f"({counter}/{total_origins})"
            )

    # Pretvaranje listi u matrice
    for horizon in horizons:

        if len(results[horizon]["dates"]) == 0:
            raise ValueError(
                f"Nisu formirane prognoze za horizont h={horizon}."
            )

        results[horizon]["dates"] = pd.DatetimeIndex(
            results[horizon]["dates"]
        )

        results[horizon]["actual"] = np.vstack(
            results[horizon]["actual"]
        )

        results[horizon]["rf"] = np.vstack(
            results[horizon]["rf"]
        )

        results[horizon]["xgb"] = np.vstack(
            results[horizon]["xgb"]
        )

    return results


# -------------------------------------------------------------------------
# 4. MODIFIKOVANI DIEBOLD-MARIANO TEST
# -------------------------------------------------------------------------
def modified_dm_test(
    loss_rf,
    loss_xgb,
    forecast_horizon=1
):

    loss_rf = np.asarray(loss_rf, dtype=float)
    loss_xgb = np.asarray(loss_xgb, dtype=float)

    if loss_rf.shape != loss_xgb.shape:
        raise ValueError(
            "Nizovi gubitaka moraju imati istu dužinu."
        )

    valid = (
        np.isfinite(loss_rf)
        & np.isfinite(loss_xgb)
    )

    loss_rf = loss_rf[valid]
    loss_xgb = loss_xgb[valid]

    differential = loss_rf - loss_xgb
    sample_size = len(differential)

    if sample_size <= forecast_horizon + 2:
        raise ValueError(
            "Broj out-of-sample prognoza je premali "
            "za izabrani prognozni horizont."
        )

    mean_differential = np.mean(differential)
    centered = differential - mean_differential

    # Kod preklapajućih h-step prognoza koriste se zaostaci
    # do h - 1.
    max_lag = min(
        forecast_horizon - 1,
        sample_size - 1
    )

    # Autokovarijansa reda 0
    long_run_variance = (
        np.dot(centered, centered) / sample_size
    )

    # Newey-West/Bartlett ponderi
    for lag in range(1, max_lag + 1):

        autocovariance = (
            np.dot(
                centered[lag:],
                centered[:-lag]
            )
            / sample_size
        )

        bartlett_weight = (
            1.0 - lag / (max_lag + 1.0)
        )

        long_run_variance += (
            2.0
            * bartlett_weight
            * autocovariance
        )

    if long_run_variance <= 0:
        return {
            "dm_statistic": np.nan,
            "p_value": np.nan,
            "mean_loss_difference": mean_differential,
            "n_forecasts": sample_size
        }

    variance_of_mean = (
        long_run_variance / sample_size
    )

    dm_original = (
        mean_differential
        / np.sqrt(variance_of_mean)
    )

    # Harvey-Leybourne-Newbold korekcija
    h = forecast_horizon
    correction_term = (
        sample_size
        + 1
        - 2 * h
        + h * (h - 1) / sample_size
    ) / sample_size

    if correction_term <= 0:
        raise ValueError(
            "HLN korekcija nije definisana za dati broj prognoza."
        )

    dm_corrected = (
        dm_original
        * np.sqrt(correction_term)
    )

    p_value = 2.0 * student_t.sf(
        np.abs(dm_corrected),
        df=sample_size - 1
    )

    return {
        "dm_statistic": dm_corrected,
        "p_value": p_value,
        "mean_loss_difference": mean_differential,
        "n_forecasts": sample_size
    }


# -------------------------------------------------------------------------
# 5. HOLM KOREKCIJA ZA 16 POJEDINAČNIH TESTOVA
# -------------------------------------------------------------------------
def holm_adjustment(p_values):
    """
    Ručno izračunavanje Holm-korigovanih p-vrednosti.
    """

    p_values = np.asarray(p_values, dtype=float)
    number_of_tests = len(p_values)

    order = np.argsort(p_values)
    adjusted = np.empty(number_of_tests)

    previous_adjusted = 0.0

    for rank, original_index in enumerate(order):

        multiplier = number_of_tests - rank

        current_adjusted = min(
            multiplier * p_values[original_index],
            1.0
        )

        current_adjusted = max(
            current_adjusted,
            previous_adjusted
        )

        adjusted[original_index] = current_adjusted
        previous_adjusted = current_adjusted

    return adjusted


# -------------------------------------------------------------------------
# 6. EVALUACIJA CELE KRIVE I POJEDINAČNIH ROČNOSTI
# -------------------------------------------------------------------------
def evaluate_dm_results(
    backtest_result,
    horizon,
    maturity_names
):
    """
    Sprovodi:
      1. DM test za celu krivu;
      2. DM test za svaku ročnost posebno.
    """

    actual = backtest_result["actual"]
    rf_prediction = backtest_result["rf"]
    xgb_prediction = backtest_result["xgb"]

    error_rf = actual - rf_prediction
    error_xgb = actual - xgb_prediction

    # Ukupni RMSE preko svih datuma i svih 16 ročnosti
    overall_rmse_rf = np.sqrt(
        np.mean(error_rf ** 2)
    )

    overall_rmse_xgb = np.sqrt(
        np.mean(error_xgb ** 2)
    )

    # Za svaki datum formira se jedna vrednost gubitka
    # kao prosečna kvadratna greška preko svih 16 ročnosti.
    curve_loss_rf = np.mean(
        error_rf ** 2,
        axis=1
    )

    curve_loss_xgb = np.mean(
        error_xgb ** 2,
        axis=1
    )

    curve_dm = modified_dm_test(
        loss_rf=curve_loss_rf,
        loss_xgb=curve_loss_xgb,
        forecast_horizon=horizon
    )

    if np.isnan(curve_dm["p_value"]):
        curve_conclusion = "Test statistic could not be calculated"

    elif curve_dm["p_value"] >= 0.05:
        curve_conclusion = (
            "No statistically significant difference"
        )

    elif curve_dm["dm_statistic"] > 0:
        curve_conclusion = (
            "XGBoost has significantly lower forecast loss"
        )

    else:
        curve_conclusion = (
            "Random Forest has significantly lower forecast loss"
        )

    summary = {
        "Horizon": horizon,
        "Number_of_forecasts": len(actual),
        "RF_RMSE": overall_rmse_rf,
        "XGBoost_RMSE": overall_rmse_xgb,
        "DM_statistic": curve_dm["dm_statistic"],
        "DM_p_value": curve_dm["p_value"],
        "Conclusion": curve_conclusion
    }

    # -------------------------------------------------------------
    # Test po pojedinačnim ročnostima
    # -------------------------------------------------------------
    maturity_rows = []

    for maturity_index, maturity_name in enumerate(maturity_names):

        loss_rf_maturity = (
            error_rf[:, maturity_index] ** 2
        )

        loss_xgb_maturity = (
            error_xgb[:, maturity_index] ** 2
        )

        maturity_dm = modified_dm_test(
            loss_rf=loss_rf_maturity,
            loss_xgb=loss_xgb_maturity,
            forecast_horizon=horizon
        )

        maturity_rows.append({
            "Maturity": maturity_name,
            "RF_RMSE": np.sqrt(
                np.mean(loss_rf_maturity)
            ),
            "XGBoost_RMSE": np.sqrt(
                np.mean(loss_xgb_maturity)
            ),
            "DM_statistic": maturity_dm["dm_statistic"],
            "p_value_raw": maturity_dm["p_value"]
        })

    maturity_table = pd.DataFrame(maturity_rows)

    maturity_table["p_value_Holm"] = holm_adjustment(
        maturity_table["p_value_raw"].values
    )

    conclusions = []

    for _, row in maturity_table.iterrows():

        if row["p_value_Holm"] >= 0.05:
            conclusion = "No significant difference"

        elif row["DM_statistic"] > 0:
            conclusion = "XGBoost"

        else:
            conclusion = "Random Forest"

        conclusions.append(conclusion)

    maturity_table["Significantly_better_model"] = conclusions

    return summary, maturity_table


# -------------------------------------------------------------------------
# 7. POKRETANJE BACKTEST-A
# -------------------------------------------------------------------------
# Januar 2018. je samo predlog.
# Možeš promeniti period, ali treba ostaviti dovoljno prognoza za DM test.


backtest_results = expanding_window_backtest(
    data=df,
    rf_model_template=rf_template,
    xgb_model_template=xgb_template,
    horizons=(1, 6),
    evaluation_start="2018-01",
    verbose=True
)


# -------------------------------------------------------------------------
# 8. DM REZULTATI ZA h=1 I h=6
# -------------------------------------------------------------------------
summary_rows = []
maturity_dm_tables = {}

for horizon in (1, 6):

    summary, maturity_table = evaluate_dm_results(
        backtest_result=backtest_results[horizon],
        horizon=horizon,
        maturity_names=list(maturity_cols)
    )

    summary_rows.append(summary)
    maturity_dm_tables[horizon] = maturity_table

    print("\n" + "=" * 80)
    print(
        f"DIEBOLD-MARIANO TEST: "
        f"RANDOM FOREST VS XGBOOST, h={horizon}"
    )
    print("=" * 80)

    print(
        f"Broj prognoza: "
        f"{summary['Number_of_forecasts']}"
    )

    print(
        f"Random Forest RMSE: "
        f"{summary['RF_RMSE']:.6f}"
    )

    print(
        f"XGBoost RMSE: "
        f"{summary['XGBoost_RMSE']:.6f}"
    )

    print(
        f"DM statistika: "
        f"{summary['DM_statistic']:.6f}"
    )

    print(
        f"p-vrednost: "
        f"{summary['DM_p_value']:.6f}"
    )

    print(
        f"Zaključak: "
        f"{summary['Conclusion']}"
    )

    print("\nRezultati po ročnostima:")
    print(
        maturity_table.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}"
        )
    )


# Sažeta tabela za celu krivu
dm_summary_df = pd.DataFrame(summary_rows)

print("\n" + "=" * 80)
print("SAŽETAK DM TESTA ZA CELU KRIVU")
print("=" * 80)

print(
    dm_summary_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}"
    )
)


with pd.ExcelWriter(
    "DM_RF_vs_XGBoost_results.xlsx",
    engine="openpyxl"
) as writer:

    dm_summary_df.to_excel(
        writer,
        sheet_name="Whole_curve",
        index=False
    )

    maturity_dm_tables[1].to_excel(
        writer,
        sheet_name="Maturities_h1",
        index=False
    )

    maturity_dm_tables[6].to_excel(
        writer,
        sheet_name="Maturities_h6",
        index=False
    )
