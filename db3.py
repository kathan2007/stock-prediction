# # ------------------------------------------------------------
# # Intraday ML Dashboard – 5‑min & 1‑h Predictions
# # v2025‑06‑22c • explicit‑column INSERT
# # ------------------------------------------------------------
# import os, sqlite3, warnings, joblib
# from datetime import datetime, time as dt_time

# import numpy as np
# import pandas as pd
# import yfinance as yf
# import ta
# from pytz import timezone
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.preprocessing import MinMaxScaler
# import streamlit as st
# from streamlit_autorefresh import st_autorefresh

# # ---------- CONFIG ----------------------------------------------------------
# FORCE_RUN  = True
# STOCKS     = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', '^NSEI', '^NSEBANK', '^BSESN']

# INTERVAL_ST, PERIOD_ST, LOOKBACK_ST = '5m',  '5d',  12   # 5‑min model
# INTERVAL_LT, PERIOD_LT, LOOKBACK_LT = '1h', '30d',  6    # 1‑hour model

# REFRESH_MS = 30_000
# MODEL_ST, SCALER_ST = 'gbr_model.pkl',      'scaler.pkl'
# MODEL_LT, SCALER_LT = 'gbr_model_long.pkl', 'scaler_long.pkl'
# DB_FILE   = os.path.join(os.getcwd(), 'signals.db')

# IST = timezone("Asia/Kolkata")
# MARKET_OPEN  = dt_time(9, 15)
# MARKET_CLOSE = dt_time(15, 30)

# warnings.filterwarnings("ignore")

# # ---------- SAFE RERUN ------------------------------------------------------
# def safe_rerun():
#     if hasattr(st,"rerun"): st.rerun()
#     elif hasattr(st,"experimental_rerun"): st.experimental_rerun()

# # ---------- DATABASE --------------------------------------------------------
# REQ_COLS = {
#     "time":"TEXT","stock":"TEXT","pred_price":"REAL",
#     "spot_price":"REAL","signal":"TEXT"
# }
# def init_db():
#     with sqlite3.connect(DB_FILE) as c:
#         c.execute("CREATE TABLE IF NOT EXISTS predictions ("+
#                   ", ".join(f"{k} {v}" for k,v in REQ_COLS.items())+")")
#         existing={r[1] for r in c.execute("PRAGMA table_info(predictions)")}
#         for col,typ in REQ_COLS.items():
#             if col not in existing:
#                 c.execute(f"ALTER TABLE predictions ADD COLUMN {col} {typ}")

# def log_row(stock,pred,spot,sig):
#     with sqlite3.connect(DB_FILE) as c:
#         c.execute("""INSERT INTO predictions
#                      (time, stock, pred_price, spot_price, signal)
#                      VALUES (?,?,?,?,?)""",
#                   (datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
#                    stock, pred, spot, sig))
#         c.commit()

# def fetch_logs(today=True, s=None):
#     sql="SELECT * FROM predictions"; flt=[]
#     if today: flt.append(f"time LIKE '{datetime.now(IST):%Y-%m-%d}%'")
#     if s: flt.append(f"stock='{s}'")
#     if flt: sql+=" WHERE "+" AND ".join(flt)
#     with sqlite3.connect(DB_FILE) as c:
#         return pd.read_sql_query(sql+" ORDER BY time", c)

# def clear_logs():
#     with sqlite3.connect(DB_FILE) as c: c.execute("DELETE FROM predictions")

# # ---------- DATA HELPERS ----------------------------------------------------
# _FIELDSET={"open","high","low","close","adj close","volume"}
# def flatten(df):
#     if not isinstance(df.columns,pd.MultiIndex): return df
#     for lvl in range(df.columns.nlevels):
#         if all(str(v).lower() in _FIELDSET for v in df.columns.get_level_values(lvl)):
#             df.columns=df.columns.get_level_values(lvl);return df
#     df.columns=df.columns.get_level_values(-1);return df

# def extract_close(df):
#     df=flatten(df)
#     for col in df.columns:
#         if "close" in col.lower(): series=df[col]; break
#     else: series=df.select_dtypes("number").iloc[:,0]
#     return pd.to_numeric(series.squeeze(), errors='coerce')

# # ---------- FEATURES --------------------------------------------------------
# FEATURES=['Close','RSI','MACD','EMA_20','Volume']
# def add_indicators(df):
#     df=flatten(df).copy()
#     df['Volume']=df.get('Volume',0)
#     close=extract_close(df)
#     df['Close']=close
#     df['RSI']=ta.momentum.RSIIndicator(close).rsi()
#     df['MACD']=ta.trend.macd_diff(close)
#     df['EMA_20']=close.ewm(span=20).mean()
#     return df.dropna()

# def make_xy(df, scaler=None, lookback=1):
#     if len(df)<=lookback:return None,None,scaler
#     X0=df[FEATURES].values
#     scaler=scaler or MinMaxScaler().fit(X0)
#     Xs=scaler.transform(X0)
#     X=np.array([Xs[i-lookback:i] for i in range(lookback,len(Xs))])
#     return X.reshape(len(X),-1),Xs[lookback:,0],scaler

# # ---------- MODEL -----------------------------------------------------------
# def train(symbol,itvl,prd,lookback,mpath,spath):
#     df=yf.download(symbol,interval=itvl,period=prd,progress=False)
#     df=add_indicators(df)
#     X,y,sc=make_xy(df,lookback=lookback)
#     reg=GradientBoostingRegressor(n_estimators=400,learning_rate=0.05).fit(X,y)
#     joblib.dump(reg,mpath); joblib.dump(sc,spath)
#     return reg,sc

# def load_or_train(symbol,itvl,prd,lookback,mpath,spath):
#     if os.path.exists(mpath) and os.path.exists(spath):
#         try:
#             reg,sc=joblib.load(mpath),joblib.load(spath)
#             if sc.n_features_in_!=len(FEATURES):
#                 raise ValueError("Feature mismatch")
#         except Exception:
#             st.warning(f"Retraining {mpath} for {len(FEATURES)}‑feature set")
#             reg,sc=train(symbol,itvl,prd,lookback,mpath,spath)
#     else:
#         reg,sc=train(symbol,itvl,prd,lookback,mpath,spath)
#     return reg,sc

# def predict(reg,sc,df_tail,lookback):
#     X,_,_=make_xy(df_tail,sc,lookback)
#     if X is None:return None
#     pscaled=reg.predict(X[-1].reshape(1,-1))[0]
#     dummy=np.zeros((1,len(FEATURES))); dummy[0,0]=pscaled
#     return sc.inverse_transform(dummy)[0,0]

# # ---------- PREDICTION ------------------------------------------------------
# def compute(sym,mst,sst,mlt,slt):
#     dfst=add_indicators(yf.download(sym,interval=INTERVAL_ST,period=PERIOD_ST,progress=False)).tail(LOOKBACK_ST+1)
#     dflt=add_indicators(yf.download(sym,interval=INTERVAL_LT,period=PERIOD_LT,progress=False)).tail(LOOKBACK_LT+1)
#     if dfst.empty or dflt.empty:return None
#     spot=float(dfst['Close'].iloc[-1])
#     pst=predict(mst,sst,dfst,LOOKBACK_ST)
#     plt=predict(mlt,slt,dflt,LOOKBACK_LT)
#     sig="HOLD"
#     if pst is not None:
#         sig="BUY" if pst>spot else "SELL" if pst<spot else "HOLD"
#         log_row(sym,pst,spot,sig)
#     return spot,pst,plt,sig

# # ---------- STREAMLIT APP ---------------------------------------------------
# def market_open(): return MARKET_OPEN <= datetime.now(IST).time() <= MARKET_CLOSE

# def main():
#     st.set_page_config(page_title="📈 Intraday ML Predictions",layout="wide")
#     st.title("📈 Intraday – 5‑min & 1‑h ML Predictions")
#     st_autorefresh(interval=REFRESH_MS,key="refresh")
#     init_db()

#     view=st.sidebar.radio("View",["All","Single"])
#     target=st.sidebar.selectbox("Stock",STOCKS) if view=="Single" else None
#     if not market_open() and not FORCE_RUN:
#         st.info("NSE closed – paused.");return

#     mst,sst=load_or_train(STOCKS[0],INTERVAL_ST,PERIOD_ST,LOOKBACK_ST,MODEL_ST,SCALER_ST)
#     mlt,slt=load_or_train(STOCKS[0],INTERVAL_LT,PERIOD_LT,LOOKBACK_LT,MODEL_LT,SCALER_LT)

#     st.markdown("---")
#     def show(sym):
#         res=compute(sym,mst,sst,mlt,slt); box=st.container()
#         if res is None: box.write("⏳ Waiting for data"); return
#         spot,pst,plt,sig=res
#         box.subheader(sym)
#         box.write(f"**Spot:** ₹{spot:.2f} | **5‑min Pred:** ₹{pst:.2f} → `{sig}` | **1‑h Pred:** ₹{plt:.2f}")

#     (show(target) if view=="Single" else [show(s) for s in STOCKS])

#     st.markdown("---")
#     st.subheader("📋 Logs (today)")
#     df=fetch_logs(True,target if view=="Single" else None)
#     st.dataframe(df if not df.empty else pd.DataFrame({"info":["No logs"]}),use_container_width=True)

#     if st.button("🧨 Clear Logs"): clear_logs(); safe_rerun()

# if __name__=="__main__":
#     main()

# ------------------------------------------------------------
# Intraday ML Dashboard – 5‑min & 1‑h Predictions + charts
# v2025‑06‑22e • chart‑index fix
# ------------------------------------------------------------
import os, sqlite3, warnings, joblib
from datetime import datetime, time as dt_time

import numpy as np
import pandas as pd
import yfinance as yf
import ta
from pytz import timezone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---------- CONFIG ----------------------------------------------------------
FORCE_RUN  = True
STOCKS     = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', '^NSEI', '^NSEBANK', '^BSESN']

INTERVAL_ST, PERIOD_ST, LOOKBACK_ST = '5m',  '5d',  12
INTERVAL_LT, PERIOD_LT, LOOKBACK_LT = '1h', '30d',  6

REFRESH_MS = 30_000
MODEL_ST, SCALER_ST = 'gbr_model.pkl',      'scaler.pkl'
MODEL_LT, SCALER_LT = 'gbr_model_long.pkl', 'scaler_long.pkl'
DB_FILE   = os.path.join(os.getcwd(), 'signals.db')

IST = timezone("Asia/Kolkata")
MARKET_OPEN  = dt_time(9, 15); MARKET_CLOSE = dt_time(15, 30)

warnings.filterwarnings("ignore")

# ---------- SAFE RERUN ------------------------------------------------------
def safe_rerun():
    if hasattr(st,"rerun"): st.rerun()
    elif hasattr(st,"experimental_rerun"): st.experimental_rerun()

# ---------- DATABASE --------------------------------------------------------
REQ_COLS = {"time":"TEXT","stock":"TEXT","pred_price":"REAL",
            "spot_price":"REAL","signal":"TEXT"}
def init_db():
    with sqlite3.connect(DB_FILE) as c:
        c.execute("CREATE TABLE IF NOT EXISTS predictions ("+
                  ", ".join(f"{k} {v}" for k,v in REQ_COLS.items())+")")
        existing={r[1] for r in c.execute("PRAGMA table_info(predictions)")}
        for col,typ in REQ_COLS.items():
            if col not in existing:
                c.execute(f"ALTER TABLE predictions ADD COLUMN {col} {typ}")

def log_row(stock,pred,spot,sig):
    with sqlite3.connect(DB_FILE) as c:
        c.execute("""INSERT INTO predictions
                     (time, stock, pred_price, spot_price, signal)
                     VALUES (?,?,?,?,?)""",
                  (datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                   stock, pred, spot, sig))
        c.commit()

def fetch_logs(today=True, s=None, limit=None):
    sql="SELECT * FROM predictions"
    flt=[]
    if today:
        flt.append(f"time LIKE '{datetime.now(IST):%Y-%m-%d}%'")
    if s:
        flt.append(f"stock='{s}'")
    if flt:
        sql += " WHERE " + " AND ".join(flt)
    sql += " ORDER BY time"
    if limit:
        sql += f" LIMIT {limit}"
    with sqlite3.connect(DB_FILE) as c:
        return pd.read_sql_query(sql, c)

def clear_logs():
    with sqlite3.connect(DB_FILE) as c:
        c.execute("DELETE FROM predictions")

# ---------- DATA HELPERS ----------------------------------------------------
_FIELDSET={"open","high","low","close","adj close","volume"}
def flatten(df):
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    for lvl in range(df.columns.nlevels):
        if all(str(v).lower() in _FIELDSET for v in df.columns.get_level_values(lvl)):
            df.columns = df.columns.get_level_values(lvl)
            return df
    df.columns = df.columns.get_level_values(-1)
    return df

def extract_close(df):
    df = flatten(df)
    for col in df.columns:
        if "close" in col.lower():
            series = df[col]
            break
    else:
        series = df.select_dtypes("number").iloc[:, 0]
    return pd.to_numeric(series.squeeze(), errors="coerce")

# ---------- FEATURES --------------------------------------------------------
FEATURES = ['Close', 'RSI', 'MACD', 'EMA_20', 'Volume']
def add_indicators(df):
    df = flatten(df).copy()
    df['Volume'] = df.get('Volume', 0)
    close = extract_close(df)
    df['Close'] = close
    df['RSI'] = ta.momentum.RSIIndicator(close).rsi()
    df['MACD'] = ta.trend.macd_diff(close)
    df['EMA_20'] = close.ewm(span=20).mean()
    return df.dropna()

def make_xy(df, scaler=None, lookback=1):
    if len(df) <= lookback:
        return None, None, scaler
    X0 = df[FEATURES].values
    scaler = scaler or MinMaxScaler().fit(X0)
    Xs = scaler.transform(X0)
    X = np.array([Xs[i - lookback:i] for i in range(lookback, len(Xs))])
    return X.reshape(len(X), -1), Xs[lookback:, 0], scaler

# ---------- MODEL -----------------------------------------------------------
def train(symbol, itvl, prd, lookback, mpath, spath):
    df = yf.download(symbol, interval=itvl, period=prd, progress=False)
    df = add_indicators(df)
    X, y, sc = make_xy(df, lookback=lookback)
    reg = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05).fit(X, y)
    joblib.dump(reg, mpath)
    joblib.dump(sc, spath)
    return reg, sc

def load_or_train(symbol, itvl, prd, lookback, mpath, spath):
    if os.path.exists(mpath) and os.path.exists(spath):
        try:
            reg, sc = joblib.load(mpath), joblib.load(spath)
            if sc.n_features_in_ != len(FEATURES):
                raise ValueError("Feature mismatch")
        except Exception:
            st.warning(f"Retraining {mpath} for {len(FEATURES)} features")
            reg, sc = train(symbol, itvl, prd, lookback, mpath, spath)
    else:
        reg, sc = train(symbol, itvl, prd, lookback, mpath, spath)
    return reg, sc

def predict(reg, sc, df_tail, lookback):
    X, _, _ = make_xy(df_tail, sc, lookback)
    if X is None:
        return None
    pscaled = reg.predict(X[-1].reshape(1, -1))[0]
    dummy = np.zeros((1, len(FEATURES)))
    dummy[0, 0] = pscaled
    return sc.inverse_transform(dummy)[0, 0]

# ---------- PREDICTION ------------------------------------------------------
def compute(sym, mst, sst, mlt, slt):
    dfst = add_indicators(
        yf.download(sym, interval=INTERVAL_ST, period=PERIOD_ST, progress=False)
    ).tail(LOOKBACK_ST + 1)
    dflt = add_indicators(
        yf.download(sym, interval=INTERVAL_LT, period=PERIOD_LT, progress=False)
    ).tail(LOOKBACK_LT + 1)
    if dfst.empty or dflt.empty:
        return None
    spot = float(dfst['Close'].iloc[-1])
    pst = predict(mst, sst, dfst, LOOKBACK_ST)
    plt = predict(mlt, slt, dflt, LOOKBACK_LT)
    sig = "BUY" if pst and pst > spot else "SELL" if pst and pst < spot else "HOLD"
    log_row(sym, pst, spot, sig)
    return spot, pst, plt, sig

# ---------- STREAMLIT APP ---------------------------------------------------
def market_open():
    return MARKET_OPEN <= datetime.now(IST).time() <= MARKET_CLOSE

def main():
    st.set_page_config(page_title="📈 Intraday ML Predictions", layout="wide")
    st.title("📈 Intraday – 5‑min & 1‑h ML Predictions + Charts")
    st_autorefresh(interval=REFRESH_MS, key="refresh")
    init_db()

    view = st.sidebar.radio("View", ["All", "Single"])
    target = st.sidebar.selectbox("Stock", STOCKS) if view == "Single" else None

    if not market_open() and not FORCE_RUN:
        st.info("NSE closed – paused.")
        return

    mst, sst = load_or_train(
        STOCKS[0], INTERVAL_ST, PERIOD_ST, LOOKBACK_ST, MODEL_ST, SCALER_ST
    )
    mlt, slt = load_or_train(
        STOCKS[0], INTERVAL_LT, PERIOD_LT, LOOKBACK_LT, MODEL_LT, SCALER_LT
    )

    st.markdown("---")

    def show(sym):
        res = compute(sym, mst, sst, mlt, slt)
        box = st.container()
        if res is None:
            box.write("⏳ Waiting for data")
            return
        spot, pst, plt, sig = res
        box.subheader(sym)
        box.write(
            f"**Spot:** ₹{spot:.2f} | **5‑min Pred:** ₹{pst:.2f} → `{sig}` | **1‑h Pred:** ₹{plt:.2f}"
        )

        # --- Chart (last 50 points) -----
        log_df = fetch_logs(today=False, s=sym, limit=50)
        if not log_df.empty:
            chart_df = (
                log_df[['time', 'spot_price', 'pred_price']]
                .rename(columns={'spot_price': 'Spot', 'pred_price': 'Pred'})
            )
            chart_df['time'] = pd.to_datetime(chart_df['time'])
            chart_df = chart_df.set_index('time').sort_index()
            box.line_chart(chart_df)

    if view == "Single":
        show(target)
    else:
        for s in STOCKS:
            show(s)

    st.markdown("---")
    st.subheader("📋 Logs (today)")
    df = fetch_logs(True, target if view == "Single" else None)
    st.dataframe(
        df if not df.empty else pd.DataFrame({"info": ["No logs"]}),
        use_container_width=True,
    )

    if st.button("🧨 Clear Logs"):
        clear_logs()
        safe_rerun()

if __name__ == "__main__":
    main()
