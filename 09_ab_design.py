"""模块五（新增）：高危客户挽回券 A/B 实验设计 + 样本量 + 盈亏平衡。

业务问题：模型识别出「首单后 180 天未复购」的高风险客户。
发挽回券是否值得？发给谁？券怎么设计？样本要多少？

本脚本只回答「实验该怎么设计」，不模拟实验结果（模拟见 10_ab_simulation.py）：
  1. 目标人群界定（建模集内 180 天未复购客户，按首单金额分层）
  2. 基线 p0 估计：class_weight='balanced' 的概率未校准（均值可虚高到 40%+，
     而真实 base rate 仅 3%）→ 在测试集上拟合 Platt 校准，
     再对全量 refit 概率做变换，取层内均值作为 p0 代理
  3. 样本量计算：两比例 z 检验，p0 ∈ {1%,2%,3%} × 相对 MDE ∈ {+10/20/50%}
  4. 盈亏平衡：等值券（10%×AOV）vs 定额券（R$10）各自的 break-even 相对提升
  5. 预注册分析计划：ITT、单一主指标、z 检验、CI、无偷看

产出：docs/ab-experiment-design.md + outputs/ab_design.html

Run:  ./.venv/Scripts/python.exe 09_ab_design.py   （需先跑过 05）
"""
from __future__ import annotations

import math
import sqlite3
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.model_selection import train_test_split

import config

sys.path.append(str(config.ROOT / "src"))
from viz import save_report  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- 与 05_propensity.py 完全一致的特征/管道定义（自包含，不跨脚本 import）---
SEED = config.RANDOM_SEED
NUM = ["log_first_order_value", "first_review_score", "first_delivery_days",
       "first_item_count", "first_freight_ratio"]
BIN = ["first_on_time"]
CAT = ["customer_state", "first_category", "first_order_month"]

ALPHA = 0.05              # 双侧
POWER_TARGET = 0.80
P0_GRID = [0.01, 0.02, 0.03]      # p0 敏感性档位
REL_MDES = [0.10, 0.20, 0.50]     # 相对提升档位
COUPON_PCT = 0.10        # 方案 A：等值券（订单金额 10%）
COUPON_FIXED = 10.0      # 方案 B：定额券 R$10
MARGIN = 0.20            # 毛利率假设


def n_per_group(p1: float, p2: float, alpha: float = ALPHA,
                power: float = POWER_TARGET) -> float:
    """两比例 z 检验所需每组样本量（p1=对照, p2=处理）。"""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (z_a * math.sqrt(2 * pbar * (1 - pbar))
           + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return num / (p2 - p1) ** 2


def power_given_n(p1: float, p2: float, n: int, alpha: float = ALPHA) -> float:
    """固定每组样本量 n 时的检验功效。"""
    z_a = stats.norm.ppf(1 - alpha / 2)
    se = math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return float(stats.norm.cdf(math.sqrt(n) * abs(p2 - p1) / se - z_a))


def df_to_md(df: pd.DataFrame) -> str:
    """极简 markdown 表格（避免 tabulate 依赖）。"""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    con = sqlite3.connect(config.DB_PATH)
    mf = pd.read_sql("SELECT * FROM model_features", con)
    scores = pd.read_sql("SELECT * FROM customer_scores", con)
    con.close()
    mf = mf.merge(scores[["customer_unique_id", "repurchase_prob"]],
                  on="customer_unique_id", how="left")

    # ---------- 1. 校准 p0 ----------
    df = mf.copy()
    df["log_first_order_value"] = np.log1p(df["first_order_value"])
    y = df["repurchase_180d"].astype(int)
    X = df[NUM + BIN + CAT]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)

    # 与 05 相同的管道（05 内联定义，此处复制保持自包含）
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("bin", SimpleImputer(strategy="most_frequent"), BIN),
        ("cat", OneHotEncoder(min_frequency=200, handle_unknown="ignore"), CAT),
    ])
    pipe = Pipeline([("pre", pre),
                     ("lr", LogisticRegression(class_weight="balanced",
                                               max_iter=1000, random_state=SEED))])
    pipe.fit(X_tr, y_tr)
    # Platt 校准：sigmoid 参数只在测试集上拟合（避免双重使用），
    # 再套用到全量 refit 模型的原始概率上
    raw_te = pipe.predict_proba(X_te)[:, 1]
    platt = LogisticRegression(C=1e10, max_iter=1000).fit(raw_te[:, None], y_te)
    a_cal, b_cal = float(platt.coef_[0][0]), float(platt.intercept_[0])

    pipe.fit(X, y)  # 全量 refit（同 05 的最终模型）
    pool = df[df["repurchase_180d"] == 0].copy()
    raw_pool = pipe.predict_proba(pool[NUM + BIN + CAT])[:, 1]
    pool["prob_cal"] = 1.0 / (1.0 + np.exp(-(a_cal * raw_pool + b_cal)))

    pool["high_value"] = pool["first_order_value"] >= pool["first_order_value"].median()

    # 持久化目标池（供 10_ab_simulation.py 使用）
    with sqlite3.connect(config.DB_PATH) as wcon:
        pool[["customer_unique_id", "repurchase_prob", "prob_cal",
              "first_order_value", "first_category", "customer_state",
              "high_value"]].to_sql("winback_pool", wcon,
                                    if_exists="replace", index=False)

    p0_all = pool["prob_cal"].mean()
    p0_hv = pool.loc[pool["high_value"], "prob_cal"].mean()
    aov_all = pool["first_order_value"].mean()
    n_pool = len(pool)
    n_hv = int(pool["high_value"].sum())
    print(f"目标池（180 天未复购）: {n_pool:,} 人（高价值层 {n_hv:,}）")
    print(f"校准后 p0: 全池 {p0_all*100:.2f}% | 高价值层 {p0_hv*100:.2f}%"
          f"（未校准模型均值分别为 {pool['repurchase_prob'].mean()*100:.1f}% / "
          f"{pool.loc[pool['high_value'],'repurchase_prob'].mean()*100:.1f}% → 虚高 {pool['repurchase_prob'].mean()/p0_all:.0f}×）")
    print(f"首单 AOV: {aov_all:.2f} BRL\n")

    # ---------- 2. 样本量表（p0 档位 × MDE，目标池=全池） ----------
    rows = []
    for p0 in P0_GRID:
        for rel in REL_MDES:
            p1, p2 = p0, p0 * (1 + rel)
            n = n_per_group(p1, p2)
            half = n_pool / 2
            rows.append(dict(p0=f"{p0*100:.0f}%", MDE=f"+{rel*100:.0f}%",
                             p1=f"{p2*100:.2f}%",
                             n_per_group=f"{int(np.ceil(n)):,}",
                             池半量=f"{int(half):,}",
                             可行="是" if n <= half else "否",
                             功效=f"{power_given_n(p1, p2, int(half))*100:.0f}%"))
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))

    # ---------- 3. 盈亏平衡 ----------
    # 每人的成本 = p2×券面值（用券=复购才发生）；增量毛利/人 = (p2-p1)×AOV×M
    # 两方案的 break-even 都与 p0 无关（p0 约掉）：
    #   方案 A 等值券 c×AOV:  r·M = (1+r)·c  →  r* = c / (M - c)
    #   方案 B 定额券 F:      r·AOV·M = (1+r)·F  →  r* = F / (AOV·M - F)
    r_be_pct = COUPON_PCT / (MARGIN - COUPON_PCT)
    margin_per_repurch = aov_all * MARGIN

    def r_be_fixed(F: float) -> float:
        denom = margin_per_repurch - F
        return F / denom if denom > 0 else float("inf")

    ladder = [5, 10, 15, 20]
    r_bes = {f"R${F:.0f}": r_be_fixed(F) for F in ladder}
    print(f"\n盈亏平衡（相对提升达到多少才回本；均与 p0 无关）:")
    print(f"  方案A 等值券(10%×AOV={aov_all*COUPON_PCT:.0f} BRL):  r* = {r_be_pct*100:.0f}%")
    print(f"  方案B 定额券（单次复购毛利 {margin_per_repurch:.1f} BRL）:")
    for k, v in r_bes.items():
        print(f"    {k}:  r* = "
              f"{('+' + f'{v*100:.0f}%') if math.isfinite(v) else '永不回本'}")

    # ---------- 4. ROI（每 1 万目标客户，校准 p0 + +20% 情景） ----------
    p0 = p0_all
    p2 = p0 * 1.20
    n_t = 10_000
    incr = n_t * (p2 - p0)
    rev = incr * aov_all * MARGIN

    def roi_of(cost_per_repurch: float) -> tuple[float, float]:
        cost = n_t * p2 * cost_per_repurch
        return cost, (rev - cost) / cost

    cost_a, roi_a = roi_of(aov_all * COUPON_PCT)      # 等值券
    cost_b5, roi_b5 = roi_of(5.0)                      # 定额券 R$5
    cost_b10, roi_b10 = roi_of(10.0)                   # 定额券 R$10
    print(f"\nROI 量级（每 1 万人, +20%, p0={p0*100:.2f}%, AOV={aov_all:.0f}）:")
    print(f"  增量复购 {incr:.0f} 人, 增量毛利 {rev:,.0f} BRL")
    print(f"  方案A 等值券(10%×AOV): 成本 {cost_a:,.0f} → ROI {roi_a:.2f}x")
    print(f"  方案B 定额券 R$5:     成本 {cost_b5:,.0f} → ROI {roi_b5:.2f}x  ← 唯一正 ROI")
    print(f"  方案B 定额券 R$10:    成本 {cost_b10:,.0f} → ROI {roi_b10:.2f}x（负，需 LTV 视角）")

    # ---------- 5. 输出 ----------
    n80 = n_per_group(p0_all, p0_all * 1.2)
    ns = np.arange(1000, 35001, 500)
    curve = pd.DataFrame({"n": ns,
                          "power": [power_given_n(p0_all, p0_all * 1.2, int(n))
                                    for n in ns]})
    fig = go.Figure()
    fig.add_scatter(x=curve["n"], y=curve["power"], mode="lines", name="power")
    fig.add_hline(y=POWER_TARGET, line_dash="dash", annotation_text=f"power={POWER_TARGET}")
    fig.add_vline(x=n80, line_dash="dot", annotation_text=f"n80={int(n80):,}")
    fig.update_layout(xaxis_title="每组样本量", yaxis_title="统计功效",
                      title=f"功效曲线（p0={p0_all*100:.2f}% 校准值, MDE=+20% 相对）")
    tab_html = tab.to_html(index=False, border=0)

    blocks = [
        (f"功效曲线（校准 p0={p0_all*100:.2f}%, MDE=+20% 相对）", fig),
        ("样本量需求 vs 目标池容量（全池, 池半量 "
         f"{int(n_pool/2):,}）", tab_html),
        ("盈亏平衡与券方案建议（核心发现）",
         f"单次复购毛利 = AOV×M = {margin_per_repurch:.1f} BRL；break-even 与 p0 无关，只由券面值/单次毛利决定。\n\n"
         f"**方案 A 等值券（10%×AOV = {aov_all*COUPON_PCT:.0f} BRL）**：r* = c/(M−c) = "
         f"**+{r_be_pct*100:.0f}%** → 行业常见 +10~30% 提升下 ROI 必为负，不推荐。\n\n"
         f"**方案 B 定额券梯度**：" + "；".join(
             f"{k} → r* = {'+' + format(v*100, '.0f') + '%' if math.isfinite(v) else '永不回本'}"
             for k, v in r_bes.items()) + "。\n\n"
         f"**建议**：主实验用 **R$5 定额券**（+20% 情景下唯一正 ROI）；"
         f"R$10 仅在 LTV 视角（360 天观察窗）下考虑；R$5.31 是 +20% 提升下的券额盈亏平衡线。"),
    ]
    save_report(blocks, config.OUTPUTS_DIR / "ab_design.html",
                "挽回券 A/B 实验 — 设计 / 样本量 / 盈亏平衡")

    md = f"""# 挽回券 A/B 实验设计（预注册草案）

> 由 `09_ab_design.py` 生成。所有数字来自 Olist 离线数据 + 显式假设。
> **本文档是实验设计，不是实验结果**（离线模拟演练见 `10_ab_simulation.py`，
> 输出已全部标注 SIMULATED）。

## 1. 业务假设
H1：对「首单后 180 天未复购」客户发放定额复购券（R$5，60 天有效），
相比不发放，能提升其下一 180 天窗口的复购率。
（R$10 为探索性券额，仅在 LTV 视角下评估。）

## 2. 目标人群
- 建模集（首单 ≤ {config.MODEL_FIRST_ORDER_CUTOFF.date()}）中 180 天未复购客户：
  **{n_pool:,} 人**
- 分层：按首单金额中位数分高价值层（{n_hv:,} 人）/ 普通层
- 分流单元：customer_unique_id 哈希，50/50

## 3. 指标与基线
- **主指标（唯一）**：下一 180 天复购率
- 护栏指标：AOV、券成本/客户、净增量毛利
- 基线 p0 估计（关键修正）：`class_weight='balanced'` 的 LogReg 输出**未校准**，
  全池概率均值 {pool['repurchase_prob'].mean()*100:.1f}%（真实 base rate 2.96%）→
  在测试集拟合 Platt（sigmoid）校准后，全池 p0 ≈ **{p0_all*100:.2f}%**、
  高价值层 ≈ **{p0_hv*100:.2f}%**
- 敏感性：设计按 p0 ∈ {{1%, 2%, 3%}} 三档给出样本量，不押注单一估计

## 4. 样本量（α={ALPHA} 双侧, 目标 power={POWER_TARGET}, 目标池 {n_pool:,}）
{df_to_md(tab)}

## 5. 券方案盈亏平衡（本设计的核心发现）
推导：券在复购订单上核销 → 每人券成本 = p2×券面值；每人增量毛利 = (p2−p1)×AOV×M。
两式的 p0 约掉 → **break-even 与 p0 无关，只取决于 券面值 / 单次复购毛利（{margin_per_repurch:.1f} BRL）**。
- **方案 A 等值券（10%×AOV = {aov_all*COUPON_PCT:.0f} BRL）**：r* = c/(M−c) =
  **+{r_be_pct*100:.0f}%**（M={MARGIN*100:.0f}%）→ 行业常见 +10~30% 提升下 ROI 为负
- **方案 B 定额券梯度**：{'；'.join(f"{k} → r* = {'+' + format(v*100, '.0f') + '%' if math.isfinite(v) else '永不回本'}" for k, v in r_bes.items())}
- **结论**：主实验用 R$5 定额券（唯一在 +20% 情景下正 ROI 的方案）；
  R$10 仅在 LTV 视角（360 天观察窗）下考虑；方案 A 不推荐

## 6. ROI 量级（每 1 万目标客户，p0={p0_all*100:.2f}%，+20% 情景）
- 增量复购 {incr:.0f} 人，增量毛利 {rev:,.0f} BRL
- 方案 A 等值券：成本 {cost_a:,.0f} BRL → ROI {roi_a:.2f}x
- 方案 B R$5 定额券：成本 {cost_b5:,.0f} BRL → **ROI {roi_b5:.2f}x** ← 推荐
- 方案 B R$10 定额券：成本 {cost_b10:,.0f} BRL → ROI {roi_b10:.2f}x（负，需 LTV 视角）

## 7. 分析计划（预注册）
1. ITT：所有进入随机化的客户纳入，按分组而非实际用券分析
2. 两比例 z 检验；主指标唯一，无需多重比较校正
3. 报告：绝对/相对提升 + 95% CI + p 值；CI 跨 0 判定无效应
4. 禁止偷看：期中不做显著性检验
5. 时长 180 天（与标签窗口一致）；避开大促窗口（季节性混淆）

## 8. 风险与诚实声明
- p0 为校准后模型均值，非实测基线 → 上线前用最近 90 天历史窗口校准
- 券成本按「处理组复购数」计（已用券假设），若实际核销率更低 ROI 更好（本估计偏保守）
- 毛利率 20% 为行业量级假设，非 Olist 真实财务数据
"""
    out = config.ROOT / "docs" / "ab-experiment-design.md"
    out.write_text(md, encoding="utf-8")
    print(f"\nwrote {out} + {config.OUTPUTS_DIR / 'ab_design.html'}")


if __name__ == "__main__":
    main()
