import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
warnings.filterwarnings("ignore", category=InterpolationWarning)
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.metrics import mean_squared_error



def data_loading(filepath):
    data = pd.read_excel(filepath, usecols="A:Q")
    yield_data = data.iloc[:, 1:].values
    maturity = data.columns[1:].str.replace("m", "").astype(int).to_numpy()

    CPI = pd.read_excel(filepath, usecols="R", header=0)
    IP = pd.read_excel(filepath, usecols="S", header=0)

    return data, yield_data, maturity, CPI, IP


"""
yield= H · βt

For each observation, we determine the β parameter using the OLS method
β= inv(HT· H)· (HT· Y) 
        """

def NS_factor_loading(la, m):
    H = np.column_stack([np.ones(len(m)),
                         (1 - np.exp(-la * m)) / (la * m),
                         ((1 - np.exp(-la * m)) / (la * m)) - np.exp(-la * m)])

    return H


def NS_ols(lambda_const, yield_data, maturity):
    H = NS_factor_loading(lambda_const, maturity)

    nobs, ncol = yield_data.shape
    rmse = np.zeros(nobs)
    beta = np.zeros((nobs, 3))
    yield_ns = np.zeros_like(yield_data)

    for t in range(nobs):
        beta_t = np.linalg.solve(np.dot(H.T, H), np.dot(H.T, yield_data[t]))
        yield_ns_t = np.dot(H, beta_t)

        beta[t, :] = beta_t
        rmse[t] = np.sqrt(np.mean((yield_data[t] - yield_ns_t) ** 2))
        yield_ns[t, :] = yield_ns_t

    return {'beta': beta, 'rmse': rmse, 'yield_ns': yield_ns}


def RNS_factor_loading(la, m):
    A = np.array([[1, 1, 0],
                  [0, -1, 0],
                  [0, 0, 1]])
    H = NS_factor_loading(la, m)
    G = np.dot(H, np.linalg.inv(A))

    return G, A


def RNS_factor_loading_tau_s(la, m, tau_s):
    exp_term = np.exp(-la * tau_s)
    A_tau_s = np.array([
        [1, (1 - exp_term) / (la * tau_s), ((1 - exp_term) / (la * tau_s)) - exp_term],
        [0, -((1 - exp_term) / (la * tau_s)), -(((1 - exp_term) / (la * tau_s)) - exp_term)],
        [0, 1 - ((1 - exp_term) / (la * tau_s)), 1 - (((1 - exp_term) / (la * tau_s)) - exp_term)]
    ])

    H = NS_factor_loading(la, m)
    G = np.dot(H, np.linalg.inv(A_tau_s))

    return G, A_tau_s


def RNS_ols(yield_data, G):
    nobs, ncol = yield_data.shape
    rmse = np.zeros(nobs)
    gamma = np.zeros((nobs, 3))
    yield_rns = np.zeros_like(yield_data)

    for t in range(nobs):
        gamma_t = np.linalg.solve(np.dot(G.T, G), np.dot(G.T, yield_data[t]))
        yield_rns_t = np.dot(G, gamma_t)

        gamma[t, :] = gamma_t
        rmse[t] = np.sqrt(np.mean((yield_data[t] - yield_rns_t) ** 2))
        yield_rns[t, :] = yield_rns_t

    return {'gamma': gamma, 'rmse': rmse, 'yield_rns': yield_rns}



def factor_stats(beta, gamma):

    factors_df = pd.DataFrame({
        'beta1': beta[:, 0],
        'beta2': beta[:, 1],
        'beta3': beta[:, 2],
        'gama1': gamma[:, 0],
        'gama2': gamma[:, 1],
        'gama3': gamma[:, 2]
    })

    dns_corr = factors_df[['beta1', 'beta2', 'beta3']].corr()
    rdns_corr = factors_df[['gama1', 'gama2', 'gama3']].corr()

    print("dns corr\n", dns_corr.round(4))
    print("rdns corr\n", rdns_corr.round(4))



def test_for_stationarity(ts):
    adf_result = adfuller(ts)
    print("ADF Statistic:", adf_result[0])
    print("p-value:", adf_result[1])
    print("Critical Values:")
    for key, value in adf_result[4].items():
        print(f"   {key}: {value}")

    if adf_result[1] < 0.05:
        print(" Serija je stacionarna (odbacuje se H0)")
    else:
        print(" Serija nije stacionarna (ne odbacuje se H0)")

    kpss_result = kpss(ts, regression='c', nlags="auto")

    print("\nKPSS Statistic:", kpss_result[0])
    print("p-value:", kpss_result[1])
    print("Critical Values:")
    for key, value in kpss_result[3].items():
        print(f"   {key}: {value}")

    if kpss_result[1] < 0.05:
        print(" Serija nije stacionarna (odbacuje se H0)")
    else:
        print(" Serija je stacionarna (ne odbacuje se H0)")



def plot_beta_factors_NS_RNS(beta, gamma, niz_datuma):
    fig, axs = plt.subplots(1, 2, figsize=(17, 7))
    niz_datuma_str = [datum.strftime("%Y-%m-%d") for datum in niz_datuma]

    axs[0].plot(beta[:, 0], lw=2, color='blue')  # Level
    axs[0].plot(beta[:, 1], lw=2, color='green')  # Slope
    axs[0].plot(beta[:, 2], lw=2, color='red')  # Curvature

    xticks = range(0, len(niz_datuma), 50)  # plotting every 11th date on the x-axis
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels([niz_datuma_str[i] for i in xticks], rotation=45, ha='right')

    axs[0].set_title(f"DNS model")
    axs[0].legend(["Level", "Slope", "Curvature"])

    axs[1].plot(gamma[:, 0], lw=2, color='blue')
    axs[1].plot(gamma[:, 1], lw=2, color='green')
    axs[1].plot(gamma[:, 2], lw=2, color='red')

    xticks = range(0, len(niz_datuma), 50)
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels([niz_datuma_str[i] for i in xticks], rotation=45, ha='right')

    axs[1].set_title(f"RDNS model")
    axs[1].legend(["Short Rate", "Slope", "Curvature"])

    plt.show()


def plot_yield_curve_two_graphs(maturity, yield_data, y1, y2, string1, string2):
    t_idx = -1
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].plot(maturity, yield_data[t_idx, :], 'o-', label='Actual Yield Curve')
    axs[0].plot(maturity, y1[t_idx, :], 'x--', label=string1)

    axs[0].set_title('Yield Curve using ' + string1)
    axs[0].set_xlabel('Maturity')
    axs[0].set_ylabel('Yield')
    axs[0].legend()

    axs[0].set_xticks(maturity)
    axs[0].set_xticklabels([f"{m}M" for m in maturity])

    axs[1].plot(maturity, yield_data[t_idx, :], 'o-', label='Actual Yield Curve')
    axs[1].plot(maturity, y2[t_idx, :], 'x--', label=string2)

    axs[1].set_title('Yield Curve using ' + string2)
    axs[1].set_xlabel('Maturity')
    axs[1].set_ylabel('Yield')
    axs[1].legend()

    axs[1].set_xticks(maturity)
    axs[1].set_xticklabels([f"{m}M" for m in maturity])

    plt.tight_layout()
    plt.show()


def plot_yield_curve_one_graph(maturity, yield_data, y1, y2, string1, string2):
    t_idx = 185
    plt.figure(figsize=(10, 5))

    plt.plot(maturity, yield_data[t_idx, :], 'o-', label='Actual Yield Curve')
    plt.plot(maturity, y1[t_idx, :], 'x--', label=string1)
    plt.plot(maturity, y2[t_idx, :], 'm--', label=string2)
    plt.title('Yield Curve using ' + string1 + " & " + string2)
    plt.xlabel('Maturity')
    plt.ylabel('Yield')

    plt.xticks(maturity, [f"{m}M" for m in maturity])

    plt.legend()
    plt.tight_layout()
    plt.show()

def calculate_rmse(actual, yield_pred, time_horizont):

    return np.sqrt(mean_squared_error(actual,yield_pred[time_horizont-1, :]))

def load_from_excel(time_horisont):
    file_path_xgb = "C:/Users/andjela.djurovic/Desktop/gradient_boosting_predictions.xlsx"
    file_path_rf = "C:/Users/andjela.djurovic/Desktop/random_forest_predictions.xlsx"

    if time_horisont==1:
        xgb_july_df = pd.read_excel(
            file_path_xgb,
            sheet_name="July_2023"
        )
        rf_july_df = pd.read_excel(
            file_path_rf,
            sheet_name="July_2023"
        )
        return xgb_july_df["XGBoost"], rf_july_df["Random_Forest"]

    elif time_horisont==6:
        xgb_dec_df = pd.read_excel(
            file_path_xgb,
            sheet_name="December_2023"
        )
        rf_dec_df = pd.read_excel(
            file_path_rf,
            sheet_name="December_2023"
        )
        return xgb_dec_df["XGBoost"], rf_dec_df["Random_Forest"]

    return 0,0



def random_forest_xgb(time_horisont):
    if time_horisont==1:
        rf = [
            4.779113,
            4.856379,
            4.872519,
            4.843939,
            4.774309,
            4.680908,
            4.580141,
            4.487464,
            4.348128,
            4.259462,
            4.165795,
            4.102291,
            4.078297,
            4.045681,
            4.022328,
            4.038428

        ]

        xgb = [
                                    5.048720,
                                    5.145234,
                                     5.169584,
                                    5.150869,
                                    5.087661,
                                    4.991082,
                                    4.880546,
                                    4.776024,
                                    4.610569,
                                    4.481701,
                                    4.303809,
                                    4.150699,
                                    4.049228,
                                    3.926172,
                                   3.754083,
                                   3.695663
        ]
    elif time_horisont == 6:
        rf = [
            4.760676,
            4.850455,
            4.871962,
            4.844827,
            4.771686,
            4.670436,
            4.559399,
            4.455967,
            4.296714,
            4.195353,
            4.085292,
            4.009607,
            3.979950,
            3.932851,
            3.893980,
            3.903754
        ]

        xgb = [
                                      5.048720,
                                     5.145234,
                                      5.169584,
                                     5.150869,
                                     5.087661,
                                     4.991082,
                                     4.880546,
                                     4.776024,
                                     4.610569,
                                     4.481701,
                                    4.303809,
                                     4.150699,
                                     4.049228,
                                    3.926172,
                                    3.754083,
                                    3.695663

        ]

    return rf, xgb





def predict(time_horizont, maturity, yield_data_test, ncol, gamma, M, N, beta, lambda_const):

    H = NS_factor_loading(lambda_const, maturity)
    G, A = RNS_factor_loading_tau_s(lambda_const, maturity, (time_horizont / 12))

    beta_pred_012 = np.zeros((time_horizont, ncol))
    beta_pred_111 = np.zeros((time_horizont, ncol))
    beta_pred_212 = np.zeros((time_horizont, ncol))

    for j in range(3):
        series = beta[:, j]

        model = SARIMAX(series, order=(0, 1, 2))
        result = model.fit(disp=False)
        forecast = result.get_forecast(steps=time_horizont)
        beta_pred_012[:, j] = forecast.predicted_mean

        model = SARIMAX(series, order=(1, 1, 1))
        result = model.fit(disp=False)
        forecast = result.get_forecast(steps=time_horizont)
        beta_pred_111[:, j] = forecast.predicted_mean

        model = SARIMAX(series, order=(2, 1, 2))
        result = model.fit(disp=False)
        forecast = result.get_forecast(steps=time_horizont)
        beta_pred_212[:, j] = forecast.predicted_mean

    yield_pred_DNS_012 = np.zeros((time_horizont, len(maturity)))
    yield_pred_DNS_111 = np.zeros((time_horizont, len(maturity)))
    yield_pred_DNS_212 = np.zeros((time_horizont, len(maturity)))

    for i in range(time_horizont):

        yield_pred_DNS_012[i] = np.dot(H, beta_pred_012[i])
        yield_pred_DNS_111[i] = np.dot(H, beta_pred_111[i])
        yield_pred_DNS_212[i] = np.dot(H, beta_pred_212[i])

    M_future = np.repeat(M[-1], time_horizont)
    N_future = np.repeat(N[-1], time_horizont)

    exog_future = np.column_stack([
        M_future,
        N_future
    ])

    gamma_df = pd.DataFrame(gamma, columns=['g1', 'g2', 'g3'])
    exog_df = pd.DataFrame({'M': M, 'N': N})

    # VARIMA(0,2) sa exog
    model_varima02 = VARMAX(gamma_df, exog=exog_df, order=(0, 2))
    result2 = model_varima02.fit(disp=False)
    gamma_pred_macro_varima_02 = result2.forecast(steps=time_horizont, exog=exog_future).values

    # VARIMA (1,1) sa exog

    model_varima = VARMAX(gamma_df, exog=exog_df, order=(1, 1))
    result = model_varima.fit(disp=False)
    gamma_pred_macro_varima_11 = result.forecast(steps=time_horizont, exog=exog_future).values

    # VARIMA(2,2) sa exog
    model_varima = VARMAX(gamma_df, exog=exog_df, order=(2, 2))
    result3 = model_varima.fit(disp=False)
    gamma_pred_macro_varima_22 = result3.forecast(steps=time_horizont, exog=exog_future).values

    # VARIMA(0,2) bez exog
    model_varima_no = VARMAX(gamma_df, order=(0, 2))
    result_no5 = model_varima_no.fit(disp=False)
    gamma_pred_varima_02 = result_no5.forecast(steps=time_horizont).values

    # VARIMA(1,1) bez exog
    model_varima_no = VARMAX(gamma_df,  order=(1, 1))
    result_no4 = model_varima_no.fit(disp=False)
    gamma_pred_varima_11 = result_no4.forecast(steps=time_horizont).values

    # VARIMA(2,2) bez exog
    model_varima_no = VARMAX(gamma_df,  order=(2, 2))
    result_no = model_varima_no.fit(disp=False)
    gamma_pred_varima_22 = result_no.forecast(steps=time_horizont).values

    yield_macro_pred_varima_11 = np.zeros((time_horizont, len(maturity)))
    yield_macro_pred_varima_02 = np.zeros((time_horizont, len(maturity)))
    yield_macro_pred_varima_22 = np.zeros((time_horizont, len(maturity)))
    yield_pred_varima_11 = np.zeros((time_horizont, len(maturity)))
    yield_pred_varima_02= np.zeros((time_horizont, len(maturity)))
    yield_pred_varima_22 = np.zeros((time_horizont, len(maturity)))

    for i in range(time_horizont):

        yield_macro_pred_varima_11[i] = np.dot(G, gamma_pred_macro_varima_11[i])
        yield_macro_pred_varima_02[i] = np.dot(G, gamma_pred_macro_varima_02[i])
        yield_macro_pred_varima_22[i] = np.dot(G, gamma_pred_macro_varima_22[i])
        yield_pred_varima_11[i] = np.dot(G, gamma_pred_varima_11[i])
        yield_pred_varima_02[i] = np.dot(G, gamma_pred_varima_02[i])
        yield_pred_varima_22[i] = np.dot(G, gamma_pred_varima_22[i])

    gamma_pred_macro_arima_111 = np.zeros((time_horizont, ncol))
    gamma_pred_macro_arima_012 = np.zeros((time_horizont, ncol))
    gamma_pred_macro_arima_212 = np.zeros((time_horizont, ncol))
    gamma_pred_arima_111 = np.zeros((time_horizont, ncol))
    gamma_pred_arima_012 = np.zeros((time_horizont, ncol))
    gamma_pred_arima_212 = np.zeros((time_horizont, ncol))

    for j in range(3):

        series = gamma[:, j]
        exog = np.column_stack([M, N])

        # --- ARIMA(0,1,2) SA EXOG ---
        model_macro = SARIMAX(series, exog=exog, order=(0, 1, 2))
        result_macro = model_macro.fit(disp=False)
        forecast_macro = result_macro.get_forecast(steps=time_horizont, exog=exog_future)
        gamma_pred_macro_arima_012[:, j] = forecast_macro.predicted_mean

        # --- ARIMA(1,1,1) SA EXOG ---
        model_macro = SARIMAX(series, exog=exog, order=(1, 1, 1))
        result_macro = model_macro.fit(disp=False)
        forecast_macro = result_macro.get_forecast(steps=time_horizont, exog=exog_future)
        gamma_pred_macro_arima_111[:, j] = forecast_macro.predicted_mean

        # --- ARIMA(2,1,2) SA EXOG ---
        model_macro = SARIMAX(series, exog=exog, order=(2, 1, 2))
        result_macro = model_macro.fit(disp=False)
        forecast_macro = result_macro.get_forecast(steps=time_horizont, exog=exog_future)
        gamma_pred_macro_arima_212[:, j] = forecast_macro.predicted_mean

        # --- ARIMA(0,1,2) BEZ EXOG ---
        model_plain = SARIMAX(series, order=(0, 1, 2))
        result_plain = model_plain.fit(disp=False)
        forecast_plain = result_plain.get_forecast(steps=time_horizont)
        gamma_pred_arima_012[:, j] = forecast_plain.predicted_mean

        # --- ARIMA(1,1,1) BEZ EXOG ---
        model_plain = SARIMAX(series, order=(1, 1, 1))
        result_plain = model_plain.fit(disp=False)
        forecast_plain = result_plain.get_forecast(steps=time_horizont)
        gamma_pred_arima_111[:, j] = forecast_plain.predicted_mean

        # --- ARIMA(2,1,2) BEZ EXOG ---
        model_plain = SARIMAX(series, order=(2, 1, 2))
        result_plain = model_plain.fit(disp=False)
        forecast_plain = result_plain.get_forecast(steps=time_horizont)
        gamma_pred_arima_212[:, j] = forecast_plain.predicted_mean


    yield_macro_pred_arima_012 = np.zeros((time_horizont, len(maturity)))
    yield_macro_pred_arima_111 = np.zeros((time_horizont, len(maturity)))
    yield_macro_pred_arima_212 = np.zeros((time_horizont, len(maturity)))
    yield_pred_arima_012 = np.zeros((time_horizont, len(maturity)))
    yield_pred_arima_111 = np.zeros((time_horizont, len(maturity)))
    yield_pred_arima_212 = np.zeros((time_horizont, len(maturity)))

    for i in range(time_horizont):

        yield_macro_pred_arima_012[i] = np.dot(G, gamma_pred_macro_arima_012[i])
        yield_macro_pred_arima_111[i] = np.dot(G, gamma_pred_macro_arima_111[i])
        yield_macro_pred_arima_212[i] = np.dot(G, gamma_pred_macro_arima_212[i])
        yield_pred_arima_012[i] = np.dot(G, gamma_pred_arima_012[i])
        yield_pred_arima_111[i] = np.dot(G, gamma_pred_arima_111[i])
        yield_pred_arima_212[i] = np.dot(G, gamma_pred_arima_212[i])

    actual = yield_data_test[time_horizont-1, :]
    rf, xgb = random_forest_xgb(time_horizont)
    rf, xgb = load_from_excel(time_horizont)

    rmse_rdns_macro_012 = calculate_rmse(actual, yield_macro_pred_arima_012, time_horizont)
    rmse_rdns_012 = calculate_rmse(actual, yield_pred_arima_012, time_horizont)
    rmse_dns_012 = calculate_rmse(actual, yield_pred_DNS_012, time_horizont)
    rmse_varima_012 = calculate_rmse(actual, yield_macro_pred_varima_02, time_horizont)
    rmse_varima_012_no = calculate_rmse(actual, yield_pred_varima_02, time_horizont)
    rmse_xgb  =  np.sqrt(mean_squared_error(actual, xgb))
    rmse_rf = np.sqrt(mean_squared_error(actual, rf))

    print("=== RMSE - 6-Month Ahead Forecast ===")
    print(f"RDNS (with macro, ARIMA(0,1,2)): {rmse_rdns_macro_012:.6f}")
    print(f"RDNS (ARIMA(0,1,2)):             {rmse_rdns_012:.6f}")
    print(f"DNS  (ARIMA(0,1,2)):             {rmse_dns_012:.6f}")
    print(f"DNS  (VARIMA + macro(0,1,2)):             {rmse_varima_012:.6f}")
    print(f"DNS  (VARIMA :             {rmse_varima_012_no:.6f}")
    print(f"RF 6 meseci :             {rmse_rf:.6f}")
    print("XGB:", rmse_xgb)

    plt.figure(figsize=(10, 5))

    plt.plot(maturity, yield_data_test[time_horizont-1, :],color='black', linewidth=1.5, label="Actual")

    # plt.plot(maturity, yield_macro_pred_arima_012[time_horizont-1, :], 'x--', color='blue', label="RDNS (with macro, ARIMA(0,1,2))")
    plt.plot(maturity, yield_pred_arima_012[time_horizont-1, :], 'x--',color='red', label="RDNS (ARIMA(0,1,2))")
    # plt.plot(maturity, yield_pred_DNS_012[time_horizont-1, :], 'x--',color='green', label="DNS (ARIMA(0,1,2))")
    # plt.plot(maturity, yield_macro_pred_varima_02[time_horizont-1, :], 'x--', label="RDNS (VARIMA(0,1,2))")


    # plt.plot(maturity, yield_macro_pred_arima_111[time_horizont-1, :], 'x--',color='blue', label="RDNS (with macro, ARIMA(1,1,1))")
    # plt.plot(maturity, yield_pred_arima_111[time_horizont-1, :], 'x--', color='cornflowerblue',label="RDNS (ARIMA(1,1,1))")
    # plt.plot(maturity, yield_pred_DNS_111[time_horizont-1, :], 'x--', color='green',label="DNS (ARIMA(1,1,1))")
    # plt.plot(maturity, yield_macro_pred_varima_11[time_horizont-1, :], 'x--',color='red', label="RDNS (with macro, VARIMA(1,1,1))")
    if time_horizont==1:
        plt.plot(maturity, yield_macro_pred_varima_11[time_horizont-1, :], 'x--',color='red', label="RDNS (with macro, VARIMA(1,1,1))")
    else:
        plt.plot(maturity, yield_pred_varima_11[time_horizont-1, :], 'x--', color='orange', label="RDNS (VARIMA(1,1,1))")


    # plt.plot(maturity, yield_macro_pred_212[time_horizont-1, :], 'x--', color='blue', label="RDNS (with macro, ARIMA(2,1,2))")
    if time_horizont==1:
        plt.plot(maturity, yield_pred_arima_212[time_horizont-1, :], 'x--', color='cornflowerblue',label="RDNS (ARIMA(2,1,2))")
    if time_horizont==6:
        plt.plot(maturity, yield_pred_DNS_212[time_horizont-1, :], 'x--', color='green',label="DNS (ARIMA(2,1,2))")
    # plt.plot(maturity, yield_macro_pred_varima_22[time_horizont-1, :], 'x--', color='red', label="RDNS (with macro, VARIMA(2,1,2))")
    # plt.plot(maturity, yield_pred_varima_22[time_horizont-1, :], 'x--', color='orange', label="RDNS (VARIMA(2,1,2))")


    plt.plot(maturity, rf, 'x--', color='cornflowerblue',label="Random Forest")
    plt.plot(maturity, xgb, 'x--', color='purple',label="Gradient Boosting")

    if time_horizont==1:
        plt.title("1-Month Ahead Yield Curve Forecast: RDNS and DNS models vs Actual yield curve (July 2023.)")
    elif time_horizont==6:
        plt.title("6-Month Ahead Yield Curve Forecast: RDNS and DNS models vs Actual yield curve (December 2023.)")
    else :
        return ;
    plt.xlabel('Maturity (Months)')
    plt.ylabel('Yield (%)')
    plt.xticks(maturity, [f"{m}M" for m in maturity])
    plt.legend()
    plt.tight_layout()
    plt.show()



def main():

    tau_s = (3 / 12)
    data, yield_data, maturity, CPI, IP = data_loading('C:/Users/andjela.djurovic/Desktop/dataset.xlsx')
    data_test, yield_data_test, maturity, CPI_test, IP_test = data_loading('C:/Users/andjela.djurovic/Desktop/test.xlsx')

    lambda_grid = np.linspace(0.01, 2, 100)
    avg_rmse = []
    for lamb in lambda_grid:
        result = NS_ols(lamb, yield_data, maturity)
        avg_rmse.append(np.mean(result['rmse']))
    best_lambda = lambda_grid[np.argmin(avg_rmse)]
    print(f"Optimal lambda: {best_lambda:.4f}")
    lambda_const = best_lambda

    out_NS = NS_ols(lambda_const, yield_data, maturity)
    beta = out_NS['beta']
    nobs, ncol = beta.shape

    if tau_s == 0:
        G, A = RNS_factor_loading(lambda_const, maturity)
    else:
        G, A = RNS_factor_loading_tau_s(lambda_const, maturity, tau_s=(3 / 12))

    out_RNS = RNS_ols(yield_data, G)
    gamma = out_RNS['gamma']
    # factor_stats(beta, gamma)

    # plot_beta_factors_NS_RNS(beta, NS_rmse, gamma, RNS_rmse, date_list)
    # plot_yield_curve_two_graphs(maturity, yield_data, yield_ns, yield_ols_RNS, "DNS model ", "RDNS model")
    # plot_yield_curve_one_graph(maturity, yield_data, yield_ns, yield_ols_RNS, "DNS model ", "RDNS model", NS_rmse, RNS_rmse)

    # test_for_stationarity(CPI)
    # test_for_stationarity(np.diff(beta[:, 2]))

    M = CPI.squeeze().to_numpy()
    N = IP.squeeze().to_numpy()

    predict(1, maturity, yield_data_test, ncol, gamma, M, N, beta, lambda_const)
    predict(6, maturity, yield_data_test, ncol, gamma, M, N, beta, lambda_const)



if __name__ == "__main__":
    main()