# -*- coding: utf-8 -*-
"""
TỐI ƯU HÓA DANH MỤC ĐẦU TƯ CỔ PHIẾU BẰNG CHIẾN LƯỢC MACD + RSI  —  WEB APP
================================================================================
App Streamlit tái hiện ĐÚNG quy trình trong tiểu luận / notebook (Nee.py):
  1) HỌC (2020): tối ưu 5 tham số/mã bằng PSO (nevergrad), chấm điểm phân vị
     lợi nhuận & Sharpe, chọn 4 mã.
  2) ĐẦU TƯ (2021–2023): chia đều vốn, đóng băng tham số, tái cân bằng theo
     tháng, giao dịch toàn bộ vào/ra, khớp tại giá mở cửa phiên kế tiếp.
  3) KIỂM ĐỊNH: t một mẫu và Wilcoxon trên lợi nhuận theo ngày.
  4) PHÂN TÍCH: so sánh mua & giữ và VN-Index, theo năm, theo từng mã.

Toàn bộ hàm tính toán giữ nguyên logic của notebook gốc nên kết quả trùng khớp:
189,90% · Sharpe 1,392 · sụt giảm −32,14% (danh mục DRH, DBC, CTS, VIX).
"""
import io
import os
import math
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st

try:
    import nevergrad as ng
    CO_NEVERGRAD = True
except Exception:
    CO_NEVERGRAD = False

# ----------------------------------------------------------------------------
# HẰNG SỐ CẤU HÌNH (theo notebook gốc)
# ----------------------------------------------------------------------------
VON_BAN_DAU         = 1_000_000_000
PHI_GIAO_DICH       = 0.0015
LAI_SUAT_PHI_RUI_RO = 0.04
SO_PHIEN_MOT_NAM    = 252
HAT_GIONG           = 42
NGAN_SACH_PSO       = 60
SO_PHIEN_KHOI_DONG  = 90
SO_MA_DANH_MUC      = 4

# Danh mục & bộ tham số đã tối ưu trong bài (để tái hiện nhanh, không cần chạy PSO)
THAM_SO_DONG_BANG = {
    "DRH": dict(macd_nhanh=14, macd_cham=60, macd_tin_hieu=6,  rsi_chu_ky=18, rsi_nguong_tren=81.32),
    "DBC": dict(macd_nhanh=20, macd_cham=49, macd_tin_hieu=7,  rsi_chu_ky=26, rsi_nguong_tren=85.00),
    "CTS": dict(macd_nhanh=14, macd_cham=28, macd_tin_hieu=6,  rsi_chu_ky=29, rsi_nguong_tren=83.97),
    "VIX": dict(macd_nhanh=6,  macd_cham=42, macd_tin_hieu=10, rsi_chu_ky=21, rsi_nguong_tren=76.78),
}
DANH_MUC_BAI = ["DRH", "DBC", "CTS", "VIX"]

# ============================================================================
# PHẦN 1 — ĐỌC & CHUẨN HOÁ DỮ LIỆU
# ============================================================================
@st.cache_data(show_spinner=False)
def doc_du_lieu(content: bytes):
    df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig", low_memory=False)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df.columns = [str(c).strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    if df["date"].isna().mean() > 0.5:                       # phòng khi định dạng ngày khác
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    for c in ["open", "high", "low", "close", "volume", "adj_open", "adj_high", "adj_low", "adj_close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)

def lay_mot_ma(df, ma):
    ma = ma.upper().strip()
    s = df[df["ticker"] == ma].copy()
    if s.empty:
        raise ValueError(f"Không tìm thấy mã {ma}")
    o = "adj_open" if "adj_open" in s.columns and s["adj_open"].notna().any() else "open"
    c = "adj_close" if "adj_close" in s.columns and s["adj_close"].notna().any() else "close"
    s["gia_mo"], s["gia_dong"] = s[o], s[c]
    s = s[["date", "gia_mo", "gia_dong"]].dropna().reset_index(drop=True)
    return s[s["gia_dong"] > 0].reset_index(drop=True)

# ============================================================================
# PHẦN 2 — CHỈ BÁO KỸ THUẬT  (giữ nguyên: RSI dùng trung bình trượt đơn giản)
# ============================================================================
def _ema(mang, chu_ky):
    return pd.Series(mang).ewm(span=chu_ky, adjust=False).mean().values

def tinh_macd(gia_dong, nhanh, cham, tin_hieu):
    duong_macd = _ema(gia_dong, nhanh) - _ema(gia_dong, cham)
    return duong_macd, _ema(duong_macd, tin_hieu)

def tinh_rsi(gia_dong, chu_ky):
    s = pd.Series(gia_dong)
    chenh = s.diff()
    tang = chenh.clip(lower=0)
    giam = -chenh.clip(upper=0)
    tb_tang = tang.rolling(window=chu_ky, min_periods=chu_ky).mean()
    tb_giam = giam.rolling(window=chu_ky, min_periods=chu_ky).mean()
    rs = tb_tang / tb_giam.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50).values

def _cat_len(a, b):
    out = np.zeros(len(a), bool); out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1]); return out

def _cat_xuong(a, b):
    out = np.zeros(len(a), bool); out[1:] = (a[1:] < b[1:]) & (a[:-1] >= b[:-1]); return out

# ============================================================================
# PHẦN 3 — TÍN HIỆU KẾT HỢP MACD + RSI
# ============================================================================
def tin_hieu_ket_hop(gia_dong, ts):
    macd, tin_hieu = tinh_macd(gia_dong, int(ts["macd_nhanh"]), int(ts["macd_cham"]), int(ts["macd_tin_hieu"]))
    rsi = tinh_rsi(gia_dong, int(ts["rsi_chu_ky"]))
    ngn = float(ts["rsi_nguong_tren"])
    mua = _cat_len(macd, tin_hieu) & (rsi < ngn)
    ban = _cat_xuong(macd, tin_hieu) | (rsi > ngn)
    return np.nan_to_num(mua).astype(bool), np.nan_to_num(ban).astype(bool)

# ============================================================================
# PHẦN 4 — BACKTEST MỘT MÃ (toàn bộ vào / ra, khớp giá mở cửa phiên kế tiếp)
# ============================================================================
def backtest_mot_ma(gia_mo, gia_dong, mua, ban, von, phi=PHI_GIAO_DICH):
    n = len(gia_dong); tien = float(von); cp = 0.0; giu = False; so_lenh = 0
    gt = np.empty(n)
    for i in range(n):
        if i > 0:
            g = gia_mo[i]
            if (not giu) and mua[i-1] and g > 0:
                cp = tien * (1 - phi) / g; tien = 0.0; giu = True; so_lenh += 1
            elif giu and ban[i-1] and g > 0:
                tien = cp * g * (1 - phi); cp = 0.0; giu = False; so_lenh += 1
        gt[i] = tien + cp * gia_dong[i]
    return gt, so_lenh

def mua_va_giu(gia_mo, gia_dong, von, phi=PHI_GIAO_DICH):
    return (von * (1 - phi)) / gia_mo[0] * gia_dong

# ============================================================================
# PHẦN 5 — CHỈ SỐ HIỆU QUẢ
# ============================================================================
def tinh_chi_so(gt, von, rf=LAI_SUAT_PHI_RUI_RO):
    eq = np.asarray(gt, float); eq = eq[~np.isnan(eq)]
    if len(eq) < 2:
        return dict(loi_nhuan_tong=np.nan, loi_nhuan_nam=np.nan, sharpe=np.nan, sut_giam_toi_da=np.nan)
    tong = eq[-1] / von - 1.0
    ret = np.diff(eq) / eq[:-1]
    n = len(eq)
    nam = (eq[-1] / von) ** (SO_PHIEN_MOT_NAM / n) - 1.0
    ex = ret - rf / SO_PHIEN_MOT_NAM
    sd = ex.std(ddof=1) if len(ex) > 1 else 0.0
    sharpe = np.sqrt(SO_PHIEN_MOT_NAM) * ex.mean() / sd if sd and sd > 1e-12 else 0.0
    mdd = (eq / np.maximum.accumulate(eq) - 1.0).min()
    return dict(loi_nhuan_tong=tong * 100, loi_nhuan_nam=nam * 100,
                sharpe=float(sharpe), sut_giam_toi_da=mdd * 100)

def loi_nhuan_theo_ngay(gt):
    eq = np.asarray(gt, float); eq = eq[~np.isnan(eq)]
    return np.diff(eq) / eq[:-1]

# --- Kiểm định bằng numpy thuần (cho kết quả trùng khớp scipy, không cần cài scipy) ---
def _betacf(a, b, x):
    EPS, FPMIN = 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c = 1.0; d = 1.0 - qab * x / qap
    d = FPMIN if abs(d) < FPMIN else d
    d = 1.0 / d; h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = FPMIN if abs(d) < FPMIN else d
        c = 1.0 + aa / c; c = FPMIN if abs(c) < FPMIN else c
        d = 1.0 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = FPMIN if abs(d) < FPMIN else d
        c = 1.0 + aa / c; c = FPMIN if abs(c) < FPMIN else c
        d = 1.0 / d; de = d * c; h *= de
        if abs(de - 1.0) < EPS:
            break
    return h

def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lb + a * math.log(x) + b * math.log(1 - x))
    return bt * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1.0 - bt * _betacf(b, a, 1 - x) / b

def _t_sf(t, dfree):                       # P(T > t) của phân phối t
    x = dfree / (dfree + t * t)
    half = 0.5 * _betai(dfree / 2.0, 0.5, x)
    return half if t >= 0 else 1.0 - half

def ttest_greater(x, mu=0.0):              # kiểm định t một mẫu, giả thuyết đối: trung bình > mu
    x = np.asarray(x, float); n = len(x)
    s = x.std(ddof=1)
    t = (x.mean() - mu) / (s / math.sqrt(n)) if s > 0 else 0.0
    return float(t), float(_t_sf(t, n - 1))

def _hang_trung_binh(a):                   # rankdata trung bình (xử lý đồng hạng)
    a = np.asarray(a, float); order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a)); ranks[order] = np.arange(1, len(a) + 1)
    sa = a[order]; i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks

def wilcoxon_greater(x, y):                # Wilcoxon dấu hạng theo cặp (xấp xỉ chuẩn, khớp scipy)
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]; n = len(d)
    if n == 0:
        return 0.0, 1.0
    r = _hang_trung_binh(np.abs(d)); rp = r[d > 0].sum()
    mn = n * (n + 1) / 4.0
    _, cnt = np.unique(np.abs(d), return_counts=True)
    tie = (cnt ** 3 - cnt).sum()
    se = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie / 48.0)
    z = (rp - mn) / se if se > 0 else 0.0
    return float(rp), float(0.5 * math.erfc(z / math.sqrt(2)))

def kiem_dinh_thong_ke(g_cl, g_bh, g_vni, rf=LAI_SUAT_PHI_RUI_RO):
    r_cl, r_bh, r_vni = loi_nhuan_theo_ngay(g_cl), loi_nhuan_theo_ngay(g_bh), loi_nhuan_theo_ngay(g_vni)
    n = min(len(r_cl), len(r_bh), len(r_vni))
    r_cl, r_bh, r_vni = r_cl[:n], r_bh[:n], r_vni[:n]
    lai = rf / SO_PHIEN_MOT_NAM
    return dict(n=int(n), t0=ttest_greater(r_cl, 0.0), te=ttest_greater(r_cl - lai, 0.0),
                wbh=wilcoxon_greater(r_cl, r_bh), wvni=wilcoxon_greater(r_cl, r_vni))

# ============================================================================
# PHẦN 6 — TỐI ƯU THAM SỐ BẰNG PSO
# ============================================================================
KHONG_GIAN = dict(macd_nhanh=(5, 20, "i"), macd_cham=(21, 60, "i"), macd_tin_hieu=(5, 15, "i"),
                  rsi_chu_ky=(5, 30, "i"), rsi_nguong_tren=(55, 85, "f"))

def _muc_tieu(gia_mo, gia_dong, ts, von):
    try:
        if int(ts["macd_nhanh"]) >= int(ts["macd_cham"]):
            return -9999.0
        mua, ban = tin_hieu_ket_hop(gia_dong, ts)
        gt, so_lenh = backtest_mot_ma(gia_mo, gia_dong, mua, ban, von)
        cs = tinh_chi_so(gt, von)
        d = cs["sharpe"]
        if not np.isfinite(d):
            return -9999.0
        if so_lenh == 0:
            d -= 5.0
        if cs["loi_nhuan_tong"] < -40:
            d -= 2.0
        return float(d)
    except Exception:
        return -9999.0

def toi_uu_tham_so(gia_mo, gia_dong, von, ngan_sach=NGAN_SACH_PSO, hat_giong=HAT_GIONG):
    """PSO bằng nevergrad (như notebook); nếu thiếu nevergrad thì dùng PSO numpy thay thế."""
    np.random.seed(hat_giong)
    if CO_NEVERGRAD:
        d = {}
        for ten, (lo, hi, kieu) in KHONG_GIAN.items():
            b = ng.p.Scalar(lower=lo, upper=hi)
            if kieu == "i":
                b = b.set_integer_casting()
            d[ten] = b
        param = ng.p.Dict(**d)
        try:
            param.random_state.seed(hat_giong)
        except Exception:
            pass
        opt = ng.optimizers.PSO(parametrization=param, budget=ngan_sach)
        best, best_d = None, -1e18
        for _ in range(ngan_sach):
            uv = opt.ask()
            val = dict(uv.value)
            d_ = _muc_tieu(gia_mo, gia_dong, val, von)
            opt.tell(uv, -d_)
            if d_ > best_d:
                best_d, best = d_, dict(val)
        return best, best_d
    # --- PSO numpy thay thế ---
    keys = list(KHONG_GIAN.keys())
    lo = np.array([KHONG_GIAN[k][0] for k in keys], float)
    hi = np.array([KHONG_GIAN[k][1] for k in keys], float)
    rng = np.random.default_rng(hat_giong)
    npart = 10
    X = rng.uniform(lo, hi, (npart, len(keys)))
    V = rng.uniform(-1, 1, (npart, len(keys))) * (hi - lo) * 0.1
    def dec(x):
        return {k: (int(round(x[i])) if KHONG_GIAN[k][2] == "i" else float(x[i])) for i, k in enumerate(keys)}
    def sc(x):
        return _muc_tieu(gia_mo, gia_dong, dec(x), von)
    pb = X.copy(); pbs = np.array([sc(x) for x in X]); gi = int(pbs.argmax())
    gb, gbs = pb[gi].copy(), pbs[gi]
    for _ in range(max(1, ngan_sach // npart)):
        r1, r2 = rng.random(X.shape), rng.random(X.shape)
        V = 0.7 * V + 1.5 * r1 * (pb - X) + 1.5 * r2 * (gb - X)
        X = np.clip(X + V, lo, hi)
        s = np.array([sc(x) for x in X]); im = s > pbs
        pb[im] = X[im]; pbs[im] = s[im]; gi = int(pbs.argmax())
        if pbs[gi] > gbs:
            gb, gbs = pb[gi].copy(), pbs[gi]
    return dec(gb), float(gbs)

# ============================================================================
# PHẦN 7 — HỌC & CHỌN DANH MỤC (2020)
# ============================================================================
def chi_so_dau_2021(df, ten_index):
    moc = lay_mot_ma(df, ten_index)
    nam = pd.to_datetime(moc["date"]).dt.year.values
    return int(np.argmax(nam >= 2021)), moc

@st.cache_data(show_spinner=False)
def chon_danh_muc(df, ten_index, ngan_sach, so_ma=SO_MA_DANH_MUC, so_lenh_min=2):
    i2021, _ = chi_so_dau_2021(df, ten_index)
    cac_ma = [t for t in sorted(df["ticker"].unique()) if t != ten_index]
    kq = []
    prog = st.progress(0.0, text="Đang tối ưu tham số từng mã trên năm 2020…")
    for j, ma in enumerate(cac_ma):
        s = lay_mot_ma(df, ma)
        if len(s) < 1000:
            prog.progress((j + 1) / len(cac_ma)); continue
        gm, gd = s["gia_mo"].values, s["gia_dong"].values
        ts, _ = toi_uu_tham_so(gm[:i2021], gd[:i2021], VON_BAN_DAU, ngan_sach)
        if ts is None:
            prog.progress((j + 1) / len(cac_ma)); continue
        mua, ban = tin_hieu_ket_hop(gd[:i2021], ts)
        gt, sl = backtest_mot_ma(gm[:i2021], gd[:i2021], mua, ban, VON_BAN_DAU)
        cs = tinh_chi_so(gt, VON_BAN_DAU)
        kq.append(dict(ma=ma, loi_nhuan_2020=cs["loi_nhuan_tong"], sharpe_2020=cs["sharpe"],
                       so_lenh_2020=sl, tham_so=ts))
        prog.progress((j + 1) / len(cac_ma))
    prog.empty()
    bang = pd.DataFrame(kq)
    hl = bang[bang["so_lenh_2020"] >= so_lenh_min].copy()
    hl["hang_loi_nhuan"] = hl["loi_nhuan_2020"].rank(pct=True)
    hl["hang_sharpe"] = hl["sharpe_2020"].rank(pct=True)
    hl["diem"] = 0.5 * hl["hang_loi_nhuan"] + 0.5 * hl["hang_sharpe"]
    hl = hl.sort_values("diem", ascending=False).reset_index(drop=True)
    return hl, hl.head(so_ma).to_dict("records")

# ============================================================================
# PHẦN 8 — ĐẦU TƯ & TÁI CÂN BẰNG
# ============================================================================
def tin_hieu_dau_tu(df, ma, ts, i2021):
    s = lay_mot_ma(df, ma)
    gm, gd = s["gia_mo"].values, s["gia_dong"].values
    bd = max(0, i2021 - SO_PHIEN_KHOI_DONG)
    mua, ban = tin_hieu_ket_hop(gd[bd:], ts)
    lui = i2021 - bd
    return gm[i2021:], gd[i2021:], mua[lui:], ban[lui:]

def chi_so_reb(ngay, tan_suat):
    if tan_suat == "khong":
        return set()
    d = pd.to_datetime(pd.Series(ngay))
    if tan_suat == "thang":
        khoa = d.dt.year * 100 + d.dt.month
    elif tan_suat == "quy":
        khoa = d.dt.year * 10 + d.dt.quarter
    elif tan_suat == "nam":
        khoa = d.dt.year
    s = set(np.where(khoa.ne(khoa.shift(1)).values)[0].tolist())
    s.discard(0)
    return s

def dung_cac_ngan(df, duoc_chon, i2021):
    ngan = []
    for r in duoc_chon:
        gm, gd, mua, ban = tin_hieu_dau_tu(df, r["ma"], r["tham_so"], i2021)
        ngan.append(dict(gia_mo=gm, gia_dong=gd, mua=mua, ban=ban, ma=r["ma"]))
    dd = min(len(n["gia_dong"]) for n in ngan)
    for n in ngan:
        for k in ["gia_mo", "gia_dong", "mua", "ban"]:
            n[k] = n[k][:dd]
    return ngan

def mo_phong_danh_muc(ngan, ti_trong, reb, phi=PHI_GIAO_DICH):
    so = len(ngan); dd = min(len(n["gia_dong"]) for n in ngan)
    tien = [ti_trong[i] * VON_BAN_DAU for i in range(so)]
    cp = [0.0] * so; giu = [False] * so; gt = np.zeros(dd); nso = 0
    for t in range(dd):
        if t > 0:
            for i in range(so):
                g = ngan[i]["gia_mo"][t]
                if (not giu[i]) and ngan[i]["mua"][t-1] and g > 0:
                    cp[i] = tien[i] * (1 - phi) / g; tien[i] = 0.0; giu[i] = True; nso += 1
                elif giu[i] and ngan[i]["ban"][t-1] and g > 0:
                    tien[i] = cp[i] * g * (1 - phi); cp[i] = 0.0; giu[i] = False; nso += 1
        if t in reb:
            gd_t = [ngan[i]["gia_dong"][t] for i in range(so)]
            gtri = [tien[i] + cp[i] * gd_t[i] for i in range(so)]
            nav = sum(gtri)
            muc_tieu = [ti_trong[i] * nav for i in range(so)]
            turnover = sum(max(0.0, muc_tieu[i] - gtri[i]) for i in range(so))
            nav2 = nav - turnover * phi
            for i in range(so):
                g = ti_trong[i] * nav2
                if giu[i]:
                    cp[i] = g / gd_t[i]; tien[i] = 0.0
                else:
                    cp[i] = 0.0; tien[i] = g
        gt[t] = sum(tien[i] + cp[i] * ngan[i]["gia_dong"][t] for i in range(so))
    return gt, nso

def danh_muc_mua_va_giu(ngan, ti_trong):
    dd = min(len(n["gia_dong"]) for n in ngan)
    gt = np.zeros(dd)
    for i, n in enumerate(ngan):
        gt += mua_va_giu(n["gia_mo"], n["gia_dong"], ti_trong[i] * VON_BAN_DAU)[:dd]
    return gt

# ============================================================================
# GIAO DIỆN STREAMLIT
# ============================================================================
st.set_page_config(page_title="Tối ưu danh mục MACD + RSI", page_icon="📈", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1180px}
[data-testid="stMetricValue"]{font-size:1.7rem}
.small{color:#6B7689;font-size:.86rem}
.box{background:#F4F6F8;border:1px solid #E3E7EF;border-radius:12px;padding:16px 18px}
.box.tq{border-left:4px solid #0E7C6B}
</style>""", unsafe_allow_html=True)

st.title("📈 Tối ưu hóa danh mục đầu tư bằng chiến lược MACD + RSI")
st.markdown('<p class="small">Kiểm định kết quả mô hình và phân tích chiến lược · '
            'HOSE · Học 2020 → Đầu tư 2021–2023 · tái hiện đúng notebook gốc</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Cấu hình")
    up = st.file_uploader("Tải tệp dữ liệu HOSE_2020_2023.csv", type=["csv"])
    che_do = st.radio("Chế độ chạy",
        ["Tái hiện nhanh (4 mã & tham số đã chọn)", "Chạy lại tối ưu hóa (PSO toàn bộ mã)"],
        help="Tái hiện nhanh dùng đúng DRH, DBC, CTS, VIX và bộ tham số đã tối ưu trong bài "
             "→ ra đúng kết quả gốc tức thì. Chế độ kia chạy lại PSO trên mọi mã rồi tự chọn 4 mã.")
    ten_index = st.text_input("Tên mã chỉ số (VN-Index)", "VNINDEX")
    rebal = st.selectbox("Tần suất tái cân bằng", ["Theo tháng", "Theo quý", "Theo năm", "Không"], 0)
    reb_map = {"Theo tháng": "thang", "Theo quý": "quy", "Theo năm": "nam", "Không": "khong"}
    ngan_sach = st.slider("Ngân sách PSO (chỉ khi chạy lại tối ưu)", 20, 120, NGAN_SACH_PSO, 10)
    alpha = st.slider("Mức ý nghĩa α (kiểm định)", 1, 10, 5, format="%d%%") / 100
    if not CO_NEVERGRAD:
        st.caption("⚠️ Chưa cài nevergrad — chế độ tối ưu dùng PSO thay thế (kết quả có thể khác đôi chút). "
                   "Thêm `nevergrad` vào requirements.txt để khớp tuyệt đối.")

TEP_DU_LIEU_MAC_DINH = "HOSE_2020_2023.csv"   # đặt cùng thư mục với app trên repo
if up is not None:
    noi_dung = up.getvalue()
elif os.path.exists(TEP_DU_LIEU_MAC_DINH):
    noi_dung = open(TEP_DU_LIEU_MAC_DINH, "rb").read()
    st.sidebar.success(f"Đang dùng dữ liệu kèm theo app: {TEP_DU_LIEU_MAC_DINH}")
else:
    st.info("⬅️ Tải tệp **HOSE_2020_2023.csv** ở thanh bên để bắt đầu. "
            "Khi deploy, hãy commit tệp dữ liệu cùng thư mục với app để app tự nạp.")
    st.markdown("""
**Quy trình app thực hiện:** học & tối ưu tham số trên 2020 → chọn 4 mã theo điểm phân vị
lợi nhuận và Sharpe → đầu tư 2021–2023 (chia đều, tái cân bằng tháng, khớp giá mở cửa phiên kế tiếp)
→ kiểm định t và Wilcoxon → phân tích so sánh với mua & giữ và VN-Index.

**Quy tắc tín hiệu** — MUA: MACD cắt lên tín hiệu *và* RSI dưới ngưỡng trên;
BÁN: MACD cắt xuống tín hiệu *hoặc* RSI vượt ngưỡng trên.
""")
    st.stop()

df = doc_du_lieu(noi_dung)
ten_index = ten_index.upper().strip()
if ten_index not in set(df["ticker"]):
    st.error(f"Không thấy mã chỉ số '{ten_index}' trong dữ liệu. Các mã ví dụ: "
             f"{', '.join(list(df['ticker'].unique())[:8])}… — hãy nhập đúng tên ở thanh bên.")
    st.stop()

i2021, moc = chi_so_dau_2021(df, ten_index)
ngay = pd.to_datetime(moc["date"]).values[i2021:]

# ---- xác định danh mục & tham số ----
rank_table = None
if che_do.startswith("Tái hiện"):
    thieu = [m for m in DANH_MUC_BAI if m not in set(df["ticker"])]
    if thieu:
        st.error(f"Thiếu mã {thieu} trong dữ liệu. Hãy dùng chế độ *Chạy lại tối ưu hóa*.")
        st.stop()
    duoc_chon = []
    for m in DANH_MUC_BAI:
        s = lay_mot_ma(df, m)
        gt20, _ = backtest_mot_ma(s["gia_mo"].values[:i2021], s["gia_dong"].values[:i2021],
                                  *tin_hieu_ket_hop(s["gia_dong"].values[:i2021], THAM_SO_DONG_BANG[m]),
                                  VON_BAN_DAU)
        duoc_chon.append(dict(ma=m, tham_so=THAM_SO_DONG_BANG[m],
                              loi_nhuan_2020=tinh_chi_so(gt20, VON_BAN_DAU)["loi_nhuan_tong"]))
else:
    with st.spinner("Đang chạy PSO trên toàn bộ mã (có thể mất 1–3 phút, kết quả được lưu lại)…"):
        rank_table, duoc_chon = chon_danh_muc(df, ten_index, ngan_sach)

ma_chon = [r["ma"] for r in duoc_chon]
ngan = dung_cac_ngan(df, duoc_chon, i2021)
tt_deu = [1.0 / len(ma_chon)] * len(ma_chon)

gt_chinh, n_reb = mo_phong_danh_muc(ngan, tt_deu, chi_so_reb(ngay, reb_map[rebal]))
cs = tinh_chi_so(gt_chinh, VON_BAN_DAU)
bh = danh_muc_mua_va_giu(ngan, tt_deu); csb = tinh_chi_so(bh, VON_BAN_DAU)
vni = lay_mot_ma(df, ten_index)
veq = mua_va_giu(vni["gia_mo"].values[i2021:][:len(gt_chinh)],
                 vni["gia_dong"].values[i2021:][:len(gt_chinh)], VON_BAN_DAU)
csv = tinh_chi_so(veq, VON_BAN_DAU)
d_ngay = pd.to_datetime(pd.Series(ngay[:len(gt_chinh)]))

t1, t2, t3, t4, t5 = st.tabs(["📊 Tổng quan", "🎯 Tối ưu & danh mục", "💰 So sánh", "🔬 Kiểm định", "🧭 Phân tích"])

with t1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lợi nhuận tổng", f"{cs['loi_nhuan_tong']:,.2f}%")
    c2.metric("Tỉ số Sharpe", f"{cs['sharpe']:,.3f}")
    c3.metric("Sụt giảm tối đa", f"{cs['sut_giam_toi_da']:,.2f}%")
    c4.metric("Giá trị cuối kỳ", f"{gt_chinh[-1]/1e9:,.3f} tỉ")
    st.caption(f"Danh mục: {', '.join(ma_chon)} · chia đều · tái cân bằng {rebal.lower()} ({n_reb} lần) · "
               f"{len(gt_chinh)-1} phiên · đầu tư {str(ngay[0])[:10]} → {str(ngay[-1])[:10]}.")
    df_eq = pd.DataFrame({"Chiến lược MACD + RSI": gt_chinh / 1e9, "Mua & giữ (danh mục)": bh / 1e9,
                          ten_index: veq / 1e9}, index=d_ngay)
    st.caption("Giá trị tài sản theo thời gian (tỉ đồng)")
    st.line_chart(df_eq, height=420, color=["#0E7C6B", "#8A93A6", "#C4CCD8"])

with t2:
    st.subheader("Bộ tham số áp dụng cho giai đoạn đầu tư")
    pdf = pd.DataFrame([{"Mã": r["ma"], "MACD nhanh": int(r["tham_so"]["macd_nhanh"]),
                         "MACD chậm": int(r["tham_so"]["macd_cham"]), "Tín hiệu": int(r["tham_so"]["macd_tin_hieu"]),
                         "Chu kỳ RSI": int(r["tham_so"]["rsi_chu_ky"]),
                         "Ngưỡng RSI": round(float(r["tham_so"]["rsi_nguong_tren"]), 2)} for r in duoc_chon]).set_index("Mã")
    st.dataframe(pdf, use_container_width=True)
    if rank_table is not None:
        st.subheader("Bảng xếp hạng năm 2020 (chọn 4 mã điểm cao nhất)")
        show = rank_table[["ma", "loi_nhuan_2020", "sharpe_2020", "so_lenh_2020", "diem"]].head(12).copy()
        show.columns = ["Mã", "Lợi nhuận 2020 %", "Sharpe 2020", "Số lệnh", "Điểm tổng hợp"]
        st.dataframe(show.round({"Lợi nhuận 2020 %": 2, "Sharpe 2020": 3, "Điểm tổng hợp": 4}).set_index("Mã"),
                     use_container_width=True)
    else:
        st.info("Chế độ **Tái hiện nhanh** — dùng đúng danh mục và tham số đã tối ưu trong bài. "
                "Chuyển sang *Chạy lại tối ưu hóa* để sinh bảng xếp hạng trực tiếp từ dữ liệu.")
    st.subheader("Quy tắc tín hiệu")
    st.markdown("""
- **MUA:** MACD cắt **lên** đường tín hiệu **và** RSI dưới ngưỡng trên.
- **BÁN:** MACD cắt **xuống** đường tín hiệu **hoặc** RSI vượt ngưỡng trên.
- Giao dịch toàn bộ vào/ra; lệnh cuối phiên khớp tại **giá mở cửa phiên kế tiếp**, phí 0,15%.
""")

with t3:
    st.subheader("So sánh với đầu tư thụ động")
    comp = pd.DataFrame({
        "Lợi nhuận %": [cs["loi_nhuan_tong"], csb["loi_nhuan_tong"], csv["loi_nhuan_tong"]],
        "Sharpe": [cs["sharpe"], csb["sharpe"], csv["sharpe"]],
        "Sụt giảm %": [cs["sut_giam_toi_da"], csb["sut_giam_toi_da"], csv["sut_giam_toi_da"]],
    }, index=["Chiến lược MACD + RSI", "Mua & giữ (4 mã)", ten_index]).round(2)
    st.dataframe(comp, use_container_width=True)
    truoc = VON_BAN_DAU; rows = []
    for nam in sorted(d_ngay.dt.year.unique()):
        vt = np.where((d_ngay.dt.year == nam).values)[0]
        if len(vt) > 1:
            cuoi = gt_chinh[vt[-1]]; rows.append((str(nam), (cuoi / truoc - 1) * 100)); truoc = cuoi
    if rows:
        yl, yv = zip(*rows)
        st.caption("Lợi nhuận theo năm (%)")
        st.bar_chart(pd.DataFrame({"Lợi nhuận %": list(yv)}, index=list(yl)), height=300, color="#0E7C6B")
    dd = lambda e: (np.asarray(e, float) / np.maximum.accumulate(np.asarray(e, float)) - 1) * 100
    st.caption("Mức sụt giảm từ đỉnh (drawdown, %)")
    st.line_chart(pd.DataFrame({"Chiến lược": dd(gt_chinh), "Mua & giữ": dd(bh)}, index=d_ngay),
                  height=300, color=["#0E7C6B", "#8A93A6"])

with t4:
    st.subheader("Kiểm định thống kê tính bền vững")
    kd = kiem_dinh_thong_ke(gt_chinh, bh, veq)
    st.caption(f"Trên chuỗi lợi nhuận theo ngày (n = {kd['n']}). Mức ý nghĩa α = {alpha*100:.0f}% (đổi ở thanh bên).")
    ten = {"t0": "t một mẫu: lợi nhuận ngày > 0", "te": "t một mẫu: vượt lãi suất phi rủi ro",
           "wbh": "Wilcoxon: chiến lược > mua & giữ", "wvni": f"Wilcoxon: chiến lược > {ten_index}"}
    rows = []
    for k in ["t0", "te", "wbh", "wvni"]:
        s, p = kd[k]
        rows.append({"Kiểm định": ten[k], "Thống kê": f"{s:,.3f}", "Giá trị p": f"{p:,.5f}",
                     "Kết luận": "✅ Có ý nghĩa" if p < alpha else "⚪ Chưa có ý nghĩa"})
    st.dataframe(pd.DataFrame(rows).set_index("Kiểm định"), use_container_width=True)
    st.markdown(f"""<div class="box tq"><b>Diễn giải.</b> Hai kiểm định t cho biết lợi nhuận theo ngày có dương
và có vượt lãi suất phi rủi ro một cách có ý nghĩa hay không. Wilcoxon so sánh trực tiếp với mua & giữ và với {ten_index}.
Nếu Wilcoxon so với mua & giữ <i>chưa</i> đạt ý nghĩa, điều đó <b>không</b> nghĩa là chiến lược kém hơn — ưu thế của
nó nằm ở <b>kiểm soát rủi ro</b> (Sharpe {cs['sharpe']:.3f} so với {csb['sharpe']:.3f}; sụt giảm
{cs['sut_giam_toi_da']:.2f}% so với {csb['sut_giam_toi_da']:.2f}%), chứ không ở lợi nhuận từng ngày cao hơn.</div>""",
                unsafe_allow_html=True)
    st.caption("Lưu ý: kiểm định t giả định lợi nhuận theo ngày độc lập; thực tế có thể có tương quan chuỗi nhẹ, "
               "nên các giá trị p mang tính tham khảo định hướng.")

with t5:
    st.subheader("Hiệu suất từng mã (giai đoạn đầu tư)")
    per = []
    for n in ngan:
        gt1, sl1 = backtest_mot_ma(n["gia_mo"], n["gia_dong"], n["mua"], n["ban"], VON_BAN_DAU / len(ma_chon))
        m1 = tinh_chi_so(gt1, VON_BAN_DAU / len(ma_chon))
        bh1 = tinh_chi_so(mua_va_giu(n["gia_mo"], n["gia_dong"], VON_BAN_DAU / len(ma_chon)), VON_BAN_DAU / len(ma_chon))
        per.append({"Mã": n["ma"], "Chiến lược %": round(m1["loi_nhuan_tong"], 2), "Sharpe": round(m1["sharpe"], 3),
                    "Sụt giảm %": round(m1["sut_giam_toi_da"], 2), "Số lệnh": sl1, "Mua & giữ %": round(bh1["loi_nhuan_tong"], 2)})
    pdf2 = pd.DataFrame(per).set_index("Mã")
    st.dataframe(pdf2, use_container_width=True)
    st.caption("Chiến lược vs mua & giữ theo từng mã (%)")
    st.bar_chart(pdf2[["Chiến lược %", "Mua & giữ %"]], height=320, color=["#0E7C6B", "#8A93A6"])

    st.subheader("So sánh tần suất tái cân bằng")
    rb = []
    for lab, f in [("Không", "khong"), ("Theo tháng", "thang"), ("Theo quý", "quy"), ("Theo năm", "nam")]:
        g2, r2 = mo_phong_danh_muc(ngan, tt_deu, chi_so_reb(ngay, f))
        m2 = tinh_chi_so(g2, VON_BAN_DAU)
        rb.append({"Tần suất": lab, "Lợi nhuận %": round(m2["loi_nhuan_tong"], 2), "Sharpe": round(m2["sharpe"], 3),
                   "Sụt giảm %": round(m2["sut_giam_toi_da"], 2), "Số lần": r2})
    st.dataframe(pd.DataFrame(rb).set_index("Tần suất"), use_container_width=True)
    st.markdown("""**Kết luận chiến lược.** Giá trị cốt lõi nằm ở **kỷ luật quản trị rủi ro** — chủ động đứng ngoài
bằng tiền mặt trong các nhịp giảm mạnh — chứ không ở dự báo chính xác từng nhịp giá. Cần đọc kết quả cùng các hạn chế:
phạm vi mã và thời gian giới hạn, danh mục chọn theo hiệu suất quá khứ, mô hình phí là một xấp xỉ; hiệu suất quá khứ
không bảo đảm tương lai.""")

st.divider()
st.caption("Tiểu luận Quản lý danh mục đầu tư · Chiến lược MACD + RSI trên HOSE · App tái hiện đúng notebook gốc.")
