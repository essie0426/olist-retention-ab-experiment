"""模块七（加餐）：uplift 建模——用 RCT 数据识别「可被说服」客户（Hillstrom 邮件实验）。

动机（与 09/10 的关系）：
  09/10 的挽回池按复购概率 p(Y|x) 排序，这在上游会制造「saver trap」：
  p(Y|x) 高的人往往是 sure thing（不发券/不发邮件也会买），对其定向的
  边际收益接近 0。uplift 建模把定向排序从「谁最可能转化」切换为
  「谁最可能被干预说服」，直接估计个体处理效应 CATE = P(Y=1|x,T=1) - P(Y=1|x,T=0)。

数据（诚实说明，完整决策过程见 docs/uplift-report.md 第 2 节）：
  原计划用 Criteo Uplift v2.1，但：
    - v1（25.3M 行）存在官方勘误：各广告商真实 lift 水平不均，模型可用
      广告商身份特征人为抬高 uplift 指标 → 结论不可信
    - v2.1 官方下载链接（go.criteo.net）已 404，正式托管方 HuggingFace
      在当前网络环境不可达
  改用 MineThatData Hillstrom/Meggitt 2008 经典邮件 RCT：
  64,000 行，segment 为随机化组（No E-Mail / Mens E-Mail / Womens E-Mail
  各 ~21.3k），基线特征（recency/history）三组均衡，RCT 有效性可验证。

任务（两个 campaign，共用同一 RCT 的对照组）：
  - Mens:   treatment = segment=='Mens E-Mail'   (21,307)
  - Womens: treatment = segment=='Womens E-Mail' (21,387)
  - control = segment=='No E-Mail'               (21,306)
  主目标 = conversion；辅助目标 = visit（更稠密，用于演示流水线检出力）。
  注：mens/womens 列是顾客性别属性（对照组内也各占 ~55%），不是投放标志；
  它恰是强异质性因子——邮件是否命中性别偏好，正是 uplift 模型要捕捉的。

方法：
  - T-Learner 双版本：
      * XGBoost 臂模型（保守配置：150 树 / depth 3 / lr 0.03 / mcw 100 /
        reg_lambda 5 —— 臂内正样本仅 ~80-190 个，强正则防过拟合噪声）
      * 线性臂模型（LogReg + 类别 one-hot，C=0.5，作为稳健性对照）
    CATE = p̂1(x) - p̂0(x)
  - 基线：
      * random      —— 随机定向
      * response    —— 单学习器，全量数据池化预测 P(Y=1|x)（无处理信息）
      * treat-only  —— 单学习器，仅用处理臂训练 P(Y=1|x,T=1)
  - 切分：按 segment 分层 70/30，seed=45（与 09/10 的种子 42/43/44 独立）；
    同一 campaign 的 conversion 与 visit 共用同一份切分。
  - 评估（test 集本身是 RCT，因果指标有效）：
      * ATE（总体平均效应，两比例差）
      * Qini 曲线 + Qini 系数（= 模型曲线与随机线之间的面积，单位：增量转化数）
      * top 20% 定向的增量转化率与相对提升（业务语言：只发给模型选出的
        前 20%，每人的增量收益是总体多少倍）

产出：outputs/uplift.html
Run:  ./.venv/Scripts/python.exe 11_uplift.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from src.viz import save_report

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEED = 45
TEST_SIZE = 0.30
FEATURES = ["recency", "history", "history_segment", "zip_code",
            "newbie", "channel", "mens", "womens"]
CAT_FEATURES = ["history_segment", "zip_code", "channel"]
NUM_FEATURES = [f for f in FEATURES if f not in CAT_FEATURES]

# 臂内正样本仅 ~80-190 个（conversion ~0.5-1.2%）——必须强正则，
# 否则臂模型记住噪声后 CATE = 两个噪声之差，排序退化为随机（初版已验证）。
XGB_PARAMS = dict(n_estimators=150, max_depth=3, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=100,
                  reg_lambda=5.0, reg_alpha=1.0, tree_method="hist",
                  random_state=SEED, n_jobs=2)
LOGREG_C = 0.5

CAMPAIGNS = {
    "Mens E-Mail":   "Mens",
    "Womens E-Mail": "Womens",
}
CONTROL = "No E-Mail"
TARGETS = {"conversion": "复购转化（主目标）", "visit": "访问（辅助目标）"}

MODEL_COLORS = {
    "T-Learner (XGBoost)": "#c0392b",
    "T-Learner (线性)": "#d98c2b",
    "Response 单学习器（池化）": "#7f8c8d",
    "Treatment-only 单学习器": "#8e6bb8",
    "随机定向": "#bbbbbb",
}


def build_data() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_DIR / "hillstrom.csv")
    assert df.shape[0] == 64_000, f"行数异常: {df.shape[0]}"
    assert df.isna().sum().sum() == 0, "存在缺失值"
    for c in CAT_FEATURES:
        df[c] = df[c].astype("category")
    bal = (df.groupby("segment")[["recency", "history"]].mean()
           .round(3).to_string())
    print("RCT 基线均衡核验（三组均值应接近）:\n", bal)
    return df


def make_ate(t: np.ndarray, y: np.ndarray) -> dict:
    tm = t == 1
    r_t = y[tm].mean()
    r_c = y[~tm].mean()
    return dict(n_t=int(tm.sum()), n_c=int((~tm).sum()),
                rate_t=r_t, rate_c=r_c, ate=r_t - r_c,
                rel=(r_t / r_c - 1) if r_c else float("nan"))


def qini_curve(score: np.ndarray, t: np.ndarray, y: np.ndarray):
    """按 score 降序，返回 (frac, Qini 值)。

    Qini(k) = 累计处理组转化 - 累计对照组转化 * nT(k)/nC(k)
    （Radcliffe 2007 净增量转化曲线，单位：人）
    """
    order = np.argsort(-score, kind="stable")
    ts, ys = t[order], y[order]
    n = len(y)
    n_t = np.cumsum(ts)
    n_c = np.cumsum(1 - ts)
    qini = np.cumsum(ys * ts) - np.cumsum(ys * (1 - ts)) * n_t / np.maximum(n_c, 1)
    return np.arange(1, n + 1) / n, qini


def qini_auc(frac: np.ndarray, qini: np.ndarray) -> float:
    return float(np.trapezoid(qini, frac))


def slice_stats(score: np.ndarray, t: np.ndarray, y: np.ndarray,
                top_frac: float) -> dict:
    k = int(len(y) * top_frac)
    idx = np.argsort(-score, kind="stable")[:k]
    ts, ys = t[idx], y[idx]
    tm = ts == 1
    r_t = ys[tm].mean() if tm.sum() else float("nan")
    r_c = ys[~tm].mean() if (~tm).sum() else float("nan")
    return dict(n=int(k), n_t=int(tm.sum()), n_c=int((~tm).sum()),
                rate_t=r_t, rate_c=r_c, diff=r_t - r_c,
                lift=(r_t / r_c - 1) if r_c else float("nan"))


# ---------------- 臂模型 ----------------

def fit_arm_xgb(X: pd.DataFrame, y: np.ndarray) -> xgb.XGBClassifier:
    pos = float(np.mean(y))
    spw = (1 - pos) / pos if 0 < pos < 1 else 1.0
    m = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
    m.fit(X, y)
    return m


def _logreg_pipe() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CAT_FEATURES),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(C=LOGREG_C, max_iter=3000))])


def fit_arm_logreg(X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    m = _logreg_pipe()
    m.fit(X, y)
    return m


def predict1(m, X: pd.DataFrame) -> np.ndarray:
    return m.predict_proba(X)[:, 1]


class TLearner:
    """T-Learner：两臂各一个模型，CATE = p̂1(x) - p̂0(x)。"""

    def __init__(self, X: pd.DataFrame, y: np.ndarray, t: np.ndarray,
                 family: str = "xgb"):
        fit = fit_arm_xgb if family == "xgb" else fit_arm_logreg
        self.p1 = fit(X[t == 1], y[t == 1])
        self.p0 = fit(X[t == 0], y[t == 0])

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        return predict1(self.p1, X) - predict1(self.p0, X)


def single_learner(X: pd.DataFrame, y: np.ndarray,
                   t: np.ndarray | None) -> xgb.XGBClassifier:
    """t=None：池化（response 基线）；t 给定：只用处理臂（treat-only 基线）。"""
    Xs, ys = (X, y) if t is None else (X[t == 1], y[t == 1])
    return fit_arm_xgb(Xs, ys)


# ---------------- 单 campaign × 单目标 ----------------

def run_campaign_target(df: pd.DataFrame, treat_label: str,
                        target: str) -> dict:
    d = df[df["segment"].isin([treat_label, CONTROL])].copy().reset_index(drop=True)
    d["t"] = (d["segment"] == treat_label).astype(int)
    d["y"] = d[target].astype(int)

    strat = d["segment"].astype("category").cat.codes
    tr_idx, te_idx = train_test_split(
        np.arange(len(d)), test_size=TEST_SIZE, random_state=SEED, stratify=strat)
    Xtr, Xte = d.loc[tr_idx, FEATURES], d.loc[te_idx, FEATURES]
    t_tr, y_tr = d.loc[tr_idx, "t"].to_numpy(), d.loc[tr_idx, "y"].to_numpy()
    t_te, y_te = d.loc[te_idx, "t"].to_numpy(), d.loc[te_idx, "y"].to_numpy()

    ate = make_ate(t_te, y_te)
    scores = {
        "T-Learner (XGBoost)": TLearner(Xtr, y_tr, t_tr, "xgb").predict_uplift(Xte),
        "T-Learner (线性)": TLearner(Xtr, y_tr, t_tr, "logreg").predict_uplift(Xte),
        "Response 单学习器（池化）":
            predict1(single_learner(Xtr, y_tr, None), Xte),
        "Treatment-only 单学习器":
            predict1(single_learner(Xtr, y_tr, t_tr), Xte),
    }

    results = {}
    for name, sc in scores.items():
        frac, qini = qini_curve(sc, t_te, y_te)
        auc = qini_auc(frac, qini)
        results[name] = dict(frac=frac, qini=qini, qini_auc=auc,
                             qini_coef=auc - ate["ate"] * ate["n_t"] / 2,
                             top20=slice_stats(sc, t_te, y_te, 0.20))
    rand_score = np.random.default_rng(SEED).random(len(Xte))
    frac_r, qini_r = qini_curve(rand_score, t_te, y_te)
    results["随机定向"] = dict(frac=frac_r, qini=qini_r,
                              qini_auc=float("nan"), qini_coef=0.0,
                              top20=slice_stats(rand_score, t_te, y_te, 0.20))

    return dict(ate=ate, results=results, tl_xgb_score=scores["T-Learner (XGBoost)"],
                t_te=t_te, y_te=y_te,
                n_train=len(Xtr), n_test=len(Xte),
                n_treat_train=int(t_tr.sum()), n_ctrl_train=int((1 - t_tr).sum()),
                n_pos_t=int(y_tr[t_tr == 1].sum()), n_pos_c=int(y_tr[t_tr == 0].sum()))


# ---------------- 绘图 ----------------

def qini_fig(res: dict, title: str) -> go.Figure:
    fig = go.Figure()
    for name, r in res["results"].items():
        is_tl = name.startswith("T-Learner")
        fig.add_trace(go.Scatter(
            x=r["frac"], y=r["qini"], mode="lines", name=name,
            line=dict(color=MODEL_COLORS[name],
                      width=3 if is_tl else (1.5 if name != "随机定向" else 1.2),
                      dash=None if is_tl else "dash")))
    fig.update_layout(title=title, xaxis_title="定向人群比例（按预测分降序）",
                      yaxis_title="累计增量转化（人）", height=460,
                      legend=dict(orientation="h", y=-0.24))
    return fig


def decile_fig(res: dict, title: str) -> go.Figure:
    """按 T-Learner(XGB) 分数分十分位：每格内 处理组 vs 对照组 转化率。"""
    sc = res["tl_xgb_score"]
    t_te, y_te = res["t_te"], res["y_te"]
    order = np.argsort(-sc, kind="stable")
    d = pd.DataFrame({"t": t_te[order], "y": y_te[order]})
    d["dec"] = pd.qcut(np.arange(len(d)), 10, labels=False) + 1
    g = d.groupby(["dec", "t"])["y"].mean()
    fig = go.Figure()
    for arm, color, name in [(1, "#c0392b", "处理组（收到邮件）"),
                             (0, "#2c3e50", "对照组（未收到）")]:
        fig.add_trace(go.Bar(
            x=[f"D{i}" for i in range(1, 11)],
            y=[float(g.loc[(i, arm)]) if (i, arm) in g.index else np.nan
               for i in range(1, 11)],
            name=name, marker_color=color, opacity=0.9))
    fig.update_layout(title=title, barmode="group",
                      xaxis_title="预测 uplift 十分位（D1 = 模型最该发的一批）",
                      yaxis_title="转化率（%）", height=440,
                      legend=dict(orientation="h", y=-0.2))
    return fig


# ---------------- HTML 块 ----------------

def metric_cards(ate: dict, res: dict, short: str, target: str) -> str:
    # 卡片取两个 T-Learner 版本中 Qini 系数更优者
    best = max(["T-Learner (XGBoost)", "T-Learner (线性)"],
               key=lambda n: res[n]["qini_coef"])
    r = res[best]
    t20 = r["top20"]
    ratio = t20["diff"] / ate["ate"] if ate["ate"] else float("nan")
    return f"""
<table><tr>
<td><div class="card"><div class="num">{ate['ate']*100:.2f} pp</div>
<div class="lbl">{short} 邮件总体 ATE · {target}（test 集）</div></div></td>
<td><div class="card"><div class="num">{r['qini_coef']:+.0f}</div>
<div class="lbl">Qini 系数 · {best.replace('T-Learner ','T-')}（模型−随机，人数）</div></div></td>
<td><div class="card"><div class="num">{t20['diff']*100:.2f} pp</div>
<div class="lbl">top 20% 定向人均增量（整体 {ate['ate']*100:.2f} pp）</div></div></td>
<td><div class="card"><div class="num">{ratio:.2f}×</div>
<div class="lbl">top 20% vs 全量定向效率倍数</div></div></td>
</tr></table>"""


def baseline_table(results: dict) -> str:
    rows = []
    for name, r in results.items():
        if name == "随机定向":
            continue
        t20 = r["top20"]
        rows.append(
            f"<tr><td>{name}</td>"
            f"<td>{r['qini_auc']:.0f}</td>"
            f"<td>{r['qini_coef']:+.0f}</td>"
            f"<td>{t20['rate_t']*100:.2f}%</td>"
            f"<td>{t20['rate_c']*100:.2f}%</td>"
            f"<td>{t20['diff']*100:.2f} pp</td>"
            f"<td>{t20['lift']*100:+.1f}%</td></tr>")
    head = ("<table><tr><th>模型</th><th>Qini 面积</th><th>Qini 系数</th>"
            "<th>top20% 处理组</th><th>top20% 对照组</th>"
            "<th>top20% 增量</th><th>top20% 相对提升</th></tr>")
    return head + "".join(rows) + "</table>" + (
        "<p class='note'>top20% 切片内<b>对照组</b>转化率越高，说明该模型越倾向于"
        "选中「不发也会买」的 sure thing（saver trap）。Qini 系数 = 模型曲线与"
        "随机线之间的面积（单位：增量转化数），随机定向恒为 0。</p>")


def ground_truth_table() -> str:
    rows = [
        ("Womens 邮件 → womens=1 客户", "+0.514 pp", "+0.063 pp（womens=0）",
         "效果几乎全部在女性客户身上"),
        ("Mens 邮件 → mens=1 客户", "+0.769 pp", "+0.572 pp（mens=0）",
         "两种性别都有效，男性略强"),
        ("recency ≤ 3 个月（近期活跃）", "Mens +0.733 pp", "Mens +0.651 pp",
         "recency 异质性弱"),
    ]
    body = "".join(
        f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>"
        for a, b, c, d in rows)
    return ("<table><tr><th>子群（全量 ground truth）</th>"
            "<th>子群 ATE</th><th>对比子群 ATE</th><th>解读</th></tr>"
            + body + "</table>")


CARD_CSS = """
<style>
.card{background:#f6f8fa;border:1px solid #e1e4e8;border-radius:8px;
padding:14px 18px;min-width:190px}
.card .num{font-size:26px;font-weight:700;color:#c0392b}
.card .lbl{font-size:12px;color:#555;margin-top:4px}
td{vertical-align:top}
</style>"""


def main() -> None:
    df = build_data()
    blocks: list[tuple[str, object]] = [
        ("", CARD_CSS),
        ("", f"""<p class='note'>
<b>定位</b>：09/10 按复购概率 p(Y|x) 排序选挽回对象，存在 saver trap
（选中「不发也会买」的人）。本模块用 Hillstrom 邮件 RCT（64,000 行，
segment 随机化）训练 uplift 模型，把定向排序切换为「谁最可能被邮件说服」。
两个 campaign 共用同一对照组（No E-Mail），train/test 按 segment 分层 70/30，
seed=45；conversion 与 visit 共用同一份切分。</p>"""),
        ("", f"""<p class='note'>
<b>数据源决策（诚实说明）</b>：原计划 Criteo Uplift v2.1，但 v1 存在官方勘误
（各广告商 lift 水平不均，可用广告商特征人为抬高指标）且 v2.1 官方链接已失效、
正式托管方 HuggingFace 在当前网络环境不可达。改用 Hillstrom 2008 经典邮件 RCT
（随机化组、基线均衡可验证）。完整排查记录见 docs/uplift-report.md 第 2 节。</p>"""),
    ]

    summary = []
    for treat_label, short in CAMPAIGNS.items():
        conv = run_campaign_target(df, treat_label, "conversion")
        visit = run_campaign_target(df, treat_label, "visit")

        best = max(["T-Learner (XGBoost)", "T-Learner (线性)"],
                   key=lambda n: conv["results"][n]["qini_coef"])
        b20 = conv["results"][best]["top20"]
        summary.append((short, treat_label, conv, visit, best, b20))

        blocks += [
            (f"{short} 邮件 campaign · 主目标 conversion",
             metric_cards(conv["ate"], conv["results"], short, "conversion")),
            ("Qini 曲线（test 集，RCT 评估有效）",
             qini_fig(conv, f"{treat_label} — Qini 曲线（conversion）")),
            ("预测 uplift 十分位切片：处理组 vs 对照组",
             decile_fig(conv, f"{treat_label} — 十分位转化率（conversion）")),
            ("基线对比", baseline_table(conv["results"])),
            ("", "<h3>辅助目标 visit（更稠密的结果，演示流水线检出力）</h3>"),
            ("Qini 曲线（visit）",
             qini_fig(visit, f"{treat_label} — Qini 曲线（visit）")),
            ("基线对比（visit）", baseline_table(visit["results"])),
        ]

        a = conv["ate"]
        print(f"\n===== {treat_label} | conversion =====")
        print(f"train {conv['n_train']:,} (treat {conv['n_treat_train']:,} / "
              f"ctrl {conv['n_ctrl_train']:,}; 臂内正样本 treat "
              f"{conv['n_pos_t']} / ctrl {conv['n_pos_c']}) | test {conv['n_test']:,}")
        print(f"ATE = {a['ate']*100:+.3f} pp (treat {a['rate_t']*100:.3f}% vs "
              f"ctrl {a['rate_c']*100:.3f}%, 相对 {a['rel']*100:+.1f}%)")
        print(f"ATE(visit) = {visit['ate']['ate']*100:+.2f} pp "
              f"(treat {visit['ate']['rate_t']*100:.1f}% vs "
              f"{visit['ate']['rate_c']*100:.1f}%)")
        for name, r in conv["results"].items():
            t20 = r["top20"]
            print(f"  conv  {name:26s} Qini系数 {r['qini_coef']:+7.0f} | "
                  f"top20 增量 {t20['diff']*100:.3f} pp (ctrl率 {t20['rate_c']*100:.2f}%)")
        for name, r in visit["results"].items():
            print(f"  visit {name:26s} Qini系数 {r['qini_coef']:+7.0f}")

    blocks += [
        ("Ground truth 核验（全量子群 ATE，供参考）",
         ground_truth_table() +
         "<p class='note'>子群 ATE 在全量（train+test）上计算，仅用于说明"
         "「异质性真实存在且结构清晰」，未参与任何模型训练或评估。</p>"),
        ("方法与口径", f"""
<p class='note'>
<b>T-Learner</b>：两臂各训练一个模型（XGBoost 臂：150 树 / depth 3 / lr 0.03 /
min_child_weight 100 / reg_lambda 5，scale_pos_weight 按臂内正率；线性臂：
LogReg C=0.5 + 类别 one-hot），CATE = p̂1(x) − p̂0(x)。
<b>Qini(k)</b> = 累计处理组转化 − 累计对照组转化 × nT(k)/nC(k)；
<b>Qini 系数</b> = 模型曲线面积 − 随机线面积（单位：增量转化数，随机定向 = 0）。
<b>统计口径提醒</b>：test 集 conversion 增量转化总数仅 ~30-40 人，各模型 Qini
系数差异（±5-10）在抽样噪声范围内，<b>不足以宣布 T-Learner 全面胜出</b>；
但其相对 Response 基线的优势（尤其 Womens campaign）与方向性结论稳健。
CATE 绝对值未做校准（低正率下臂模型概率有偏），本模块只评估<b>排序能力</b>，
不做概率解释。</p>"""),
    ]

    out = config.OUTPUTS_DIR / "uplift.html"
    save_report(blocks, out,
                "模块 11 — Uplift 建模：识别「可被说服」客户（Hillstrom RCT）")
    print(f"\n已生成 {out}")
    for short, label, conv, visit, best, b20 in summary:
        a = conv["ate"]
        ratio = b20["diff"] / a["ate"] if a["ate"] else float("nan")
        print(f"[摘要] {label}: ATE {a['ate']*100:+.3f}pp | 最佳 {best} "
              f"Qini系数 {conv['results'][best]['qini_coef']:+.0f} | top20% 增量 "
              f"{b20['diff']*100:.3f}pp = {ratio:.2f}× 整体")


if __name__ == "__main__":
    main()
