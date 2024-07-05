import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from itertools import product
from sklearn.model_selection import GridSearchCV
from matplotlib.dates import DateFormatter

pd.set_option("display.width", 1500)
pd.set_option("display.max_columns", None)
#########################################################################################################################################################################################################
df = pd.read_excel(r"C:\Users\ozdmr\OneDrive\Masaüstü\EpiaşForecast\Datasets\Gerçek_Zamanlı_Tüketim_Epias.xlsx", parse_dates=["Tarih"])
val_df = pd.read_excel(r"C:\Users\ozdmr\OneDrive\Masaüstü\EpiaşForecast\Datasets\Validation2.xlsx", parse_dates=["Tarih"])
val_df1 = pd.concat([df,val_df], axis=0)
#########################################################################################################################################################################################################
def işlemler(df):
    df["nciSaat"] = df["Saat"].apply(lambda x: (int(str(x).split(":")[0]) + 1 ) if pd.notna(x) else x)
    df["SadeceTarih"] = df["Tarih"].dt.date
    df.rename(columns={"Tüketim Miktarı(MWh)": "TuketimMWh"}, inplace=True)
    df["SadeceTarih"] = pd.to_datetime(df["SadeceTarih"])

    df["Ay"] = df["SadeceTarih"].dt.month
    df["AyinGunu"] = df["SadeceTarih"].dt.day
    df["Yil"] = df["SadeceTarih"].dt.year
    df['HaftaninGunu'] = df["SadeceTarih"].dt.dayofweek + 1
    df["HaftaSonuMu"] = df["SadeceTarih"].dt.weekday // 4
    df['AyBasiMi'] = df["SadeceTarih"].dt.is_month_start.astype(int)
    df['AySonuMu'] = df["SadeceTarih"].dt.is_month_end.astype(int)

    Kurban2022 = pd.date_range(start='7/9/22', end='7/12/22', freq='D')
    KurbanArefe2022 = pd.date_range(start='7/6/22', end='7/8/22', freq='D')
    Kurban2023 = pd.date_range(start='6/28/23', end='7/1/23', freq='D')
    KurbanArefe2023 = pd.date_range(start='6/25/23', end='6/27/23', freq='D')
    Kurban2024 = pd.date_range(start='6/16/24', end='6/19/24', freq='D')
    KurbanArefe2024 = pd.date_range(start='6/13/24', end='6/15/24', freq='D')

    KurbanBayramları = Kurban2024.union(Kurban2023).union(Kurban2022)
    KurbanArefeler = KurbanArefe2024.union(KurbanArefe2023).union(KurbanArefe2022)
    df["KurbanArefe"] = 0
    df["KurbanMi"] = 0
    df.loc[df["SadeceTarih"].isin(KurbanBayramları), "KurbanMi"] = 1
    df.loc[df["SadeceTarih"].isin(KurbanArefeler), "KurbanArefe"] = 1
    df["KurbanMi"].value_counts()
    df["KurbanArefe"].value_counts()

    Ramazan2022 = pd.date_range(start='5/2/22', end='5/4/22', freq='D')
    RamazanArefe2022 = pd.date_range(start='4/29/22', end='5/1/22', freq='D')
    Ramazan2023 = pd.date_range(start='4/21/23', end='4/23/23', freq='D')
    RamazanArefe2023 = pd.date_range(start='4/18/23', end='4/20/23', freq='D')
    Ramazan2024 = pd.date_range(start='4/10/24', end='4/12/24', freq='D')
    RamazanArefe2024 = pd.date_range(start='4/7/24', end='4/9/24', freq='D')
    RamazanBayramları = Ramazan2024.union(Ramazan2023).union(Ramazan2022)
    RamazanArefeler = RamazanArefe2024.union(RamazanArefe2023).union(RamazanArefe2022)
    df["RamazanMi"] = 0
    df["RamazanArefe"] = 0
    df.loc[df["SadeceTarih"].isin(RamazanBayramları), "RamazanMi"] = 1
    df.loc[df["SadeceTarih"].isin(RamazanArefeler), "RamazanArefe"] = 1
    df["RamazanMi"].value_counts()
    df["RamazanArefe"].value_counts()
    df.head()
    df["YilBasi"] = 0
    df.loc[df["SadeceTarih"].isin(['2022-01-01', '2023-01-01', '2024-01-01']), "YilBasi"] = 1
    df["YilBasi"].value_counts()
    df["Nisan23"] = 0
    df.loc[df["SadeceTarih"].isin(['2022-04-23', '2023-04-23', '2024-04-23']), "Nisan23"] = 1
    df["Nisan23"].value_counts()
    df["Mayis1"] = 0
    df.loc[df["SadeceTarih"].isin(['2022-05-01', '2023-05-01', '2024-05-01']), "Mayis1"] = 1
    df["Mayis1"].value_counts()
    df["Mayis19"] = 0
    df.loc[df["SadeceTarih"].isin(['2022-05-19', '2023-05-19', '2024-05-19']), "Mayis19"] = 1
    df["Mayis19"].value_counts()
    df["Temmuz15"] = 0
    df.loc[df["SadeceTarih"].isin(['2022-07-15', '2023-07-15', '2024-07-15']), "Temmuz15"] = 1
    df["Temmuz15"].value_counts()
    df["Agustos30"] = 0
    df.loc[df["SadeceTarih"].isin(['2022-08-30', '2023-08-30', '2024-08-30']), "Agustos30"] = 1
    df["Agustos30"].value_counts()
    df["Ekim29"] = 0
    df.loc[df["SadeceTarih"].isin(['2022-10-29', '2023-10-29', '2024-10-29']), "Ekim29"] = 1
    df["Ekim29"].value_counts()

    df["Range"] = df.groupby("SadeceTarih")["TuketimMWh"].transform(lambda x: x.max() - x.min())
    labels = [1, 2, 3, 4, 5]
    bins = [0, 5000, 8000, 12000, 16000, 21000]
    df["Range_binned"] = pd.cut(df["Range"], bins=bins, labels=labels)
    df["Range_binned"].shift(1)
    df["Range_binned"] = df["Range_binned"].shift(1)

    # def random_noise(dataframe):
    #     return np.random.normal(scale=1.6, size=(len(dataframe),))

    # def lag_fonksiyonu(df, lag_count):
    #     for lag in lag_count:
    #         df["TuketimLag" + str(lag)] = df["TuketimMWh"].shift(lag) + random_noise(df)
    #     return df
    #
    # # lag_df = lag_fonksiyonu(df,
    # #                         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 48,
    # #                          72, 96, 120, 144, 168, 192])
    #
    # def roll_mean(df, windows):
    #     for window in windows:
    #         df["TuketimRoll" + str(window)] = df["TuketimMWh"].shift(1).rolling(window=window).mean() + random_noise(df)
    #     return df
    #
    # # lag_roll_df = roll_mean(df,
    # #                         [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 48, 72,
    # #                          96, 120, 144, 168, 192])

    def ewm_fonksiyou(df, lags, alphas):
        df_copy = df.copy()
        for alpha in alphas:
            for lag in lags:
                df_copy['TuketimMWh_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = df_copy[
                    "TuketimMWh"].shift(lag).ewm(alpha=alpha).mean()
        return df_copy

    lag_roll_ewm_df = ewm_fonksiyou(df,
                                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                     24, 48, 72, 96, 120, 144, 168, 192], [0.95, 0.9, 0.8])
    lag_roll_ewm_df.drop(["Tarih", "Saat", "AySonuMu", "Yil", "Range"], axis=1, inplace=True)
    df_final = pd.get_dummies(lag_roll_ewm_df, columns=[ 'nciSaat','Ay', 'AyinGunu', 'HaftaninGunu'], drop_first=True)

    return df_final

#########################################################################################################################################################################################################

df_final = işlemler(df)
val_df1 = işlemler(val_df1)
val_df1 = val_df1.iloc[val_df1.shape[0]-48:val_df1.shape[0]]

train = df_final.loc[(df_final["SadeceTarih"] < "2024-01-24")]
test = df_final.loc[(df_final["SadeceTarih"] >= "2024-01-24")]

cols = [col for col in train.columns if col not in ["TuketimMWh", "SadeceTarih"]]
trainX = train[cols]
trainY = train["TuketimMWh"]

testX = test[cols]
testY = test["TuketimMWh"]

valX = val_df1[cols]
valY = val_df1["TuketimMWh"]
# LightGBM veri seti
train_data = lgb.Dataset(trainX, label=trainY)
test_data = lgb.Dataset(testX, label=testY, reference=train_data)

# LightGBM parametreleri
params = {
    'objective': 'regression',
    'boosting_type': 'gbdt',
    'metric': 'mae',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.9,
    'num_boost_round':10000,
    'early_stopping_rounds': 50,
     'verbose_eval': 100
}
model = lgb.train(train_set=train_data,
                  params=params,
                  valid_sets=[train_data, test_data],
                  verbose_eval = 100)
# param_grid = {
#     'num_leaves': [31, 50, 70],
#     'max_depth': [-1, 10, 20],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'num_boost_round': [5000, 7000, 10000],
# }
#
# # Model eğitimi
#
#
# param_combinations = list(product(*param_grid.values()))
# param_names = list(param_grid.keys())
#
# best_score = 0
# best_params = None
# for params in param_combinations:
#     param_dict = dict(zip(param_names, params))
#     param_dict['objective'] = 'regression'
#     param_dict['metric'] = 'l1'
#
#     bst = lgb.train(
#         param_dict,
#         train_data,
#         num_boost_round=param_dict['num_boost_round'],
#         valid_sets=[train_data, test_data],
#         early_stopping_rounds=50,
#         verbose_eval=100
#     )
#
#     y_pred = bst.predict(testX, num_iteration=bst.best_iteration)
#     mae = mean_absolute_error(testY, y_pred)
#
#
#
#     if mae > best_score:
#         best_score = mae
#         best_params = param_dict
#
# best_score
# best_params
def smape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100


y_pred = model.predict(testX, num_iteration=model.best_iteration)
mae = mean_absolute_error(testY, y_pred)
smape = smape(testY, y_pred)
print(f'Mean Absolute Error: {mae}, Smape: {smape}') # Mean Absolute Error: 367.16754074247257, Smape: 1.018011842821955

pred_val = model.predict(valX, num_iteration=model.best_iteration)
mae_val = mean_absolute_error(valY, pred_val)
smape_val = smape(valY, pred_val)
print(f'Mean Absolute Error: {mae_val}, Smape: {smape_val}') # Mean Absolute Error: 609.8508018290448, Smape: 1.4150301501440836
# valY.mean(), valY.std()

plt.figure(figsize=(10, 5))
plt.plot(valX.index, valY, label='Gerçek')
plt.plot(valX.index, pred_val, label='Tahmin')
plt.legend()
plt.title(f'MAE: {int(mae_val)}, Ortalama Tüketim: {int(valY.mean())}')
plt.ylim(0, 60000)
plt.show(block=True)


# plt.figure(figsize=(10, 5))
# plt.plot(test.index[3232:3672], testY.iloc[3232:3672], label='Gerçek')
# plt.plot(test.index[3232:3672], y_pred[3232:3672], label='Tahmin')
# plt.legend()
# plt.title('Gerçek ve Tahmin Değerleri')
# plt.show(block=True)

# def plot_lgb_importances(model, plot=False, num=10):
#     gain = model.feature_importance('gain')
#     feat_imp = pd.DataFrame({'feature': model.feature_name(),
#                              'split': model.feature_importance('split'),
#                              'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
#     if plot:
#         plt.figure(figsize=(10, 10))
#         sns.set(font_scale=1)
#         sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
#         plt.title('feature')
#         plt.tight_layout()
#         plt.show(block=True)
#     else:
#         print(feat_imp.head(num))
#     return feat_imp
#
# res = plot_lgb_importances(model, num=200, plot=False)
# res[:30]

