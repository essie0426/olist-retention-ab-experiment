"""模块六（新增）：挽回券 A/B 离线模拟演练（SIMULATED，非真实实验）。

诚实定位：Olist 是历史交易数据，没有真实的「发券/不发券」随机化，
无法估计真实因果效应。本模块的价值是：
  1. 上线前把预注册分析流水线（09 设计的第 7 节）完整跑通一遍
  2. 用多档注入效应（0/10/20/50%）验证「流水线能否检出预期效应」
  3. 输出分品类 × 价值层的策略矩阵（哪些层值得发 R$5 券）

方法：
  - 目标池 = 09 生成的 winback_pool 表（64,741 个 180 天未复购客户）
  - 分流：rng(seed=43) 50/50 随机（与建模种子 42 不同，避免相关）
  - 注入：处理组 p_i' = p_i × (1+uplift)，uplift 为档位假设值；
    对照 p_i = prob_cal_i（Platt 校准后的模型概率）
  - 结果：outcome ~ Bernoulli(p) ，rng(seed=44)
  - 分析：按 09 预注册计划执行（两比例 z 检验 + 95% CI + ITT）
  - ROI：R$5 定额券，成本 = 处理组复购数 × 5；增量毛利 = Δ复购 × AOV × 20%

产出：outputs/ab_simulation.html

Run:  ./.venv/Scripts/python.exe 10_ab_simulation.py   （需先跑过 09）
"""
from __future__ import annotations

import sqlite3
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

import config

sys.path.append(str(config.ROOT / "src"))
from viz import save_report  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEED_ASSIGN = 43          # 分流种子（≠ 建模种子 42）
SEED_OUTCOME = 44         # 结果抽样种子
COUPON = 5.0              # 与 09 主方案一致：R$5 定额券
MARGIN = 0.20
UPLIFT_GRID = [0.0, 0.10, 0.20, 0.50]   # 注入档位（0 = 负对照）
ALPHA = 0.05


def run_analysis(rate_t: float, n_t: int, rate_c: float, n_c: int) -> dict:
    """预注册分析：两比例 z 检验 + 95% CI（与 09 第 7 节一致）。"""
    p_pool = (n_t * rate_t + n_c * rate_c) / (n_t + n_c)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
    diff = rate_t - rate_c
    z = diff / se if se > 0 else float("nan")
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    ci = (diff - 1.96 * se, diff + 1.96 * se)
    return dict(rate_t=rate_t, rate_c=rate_c, diff=diff, rel=diff / rate_c
                if rate_c else float("nan"), z=z, p=p, ci=ci,
                sig="显著" if p < ALPHA else "不显著")


def main() -> None:
    con = sqlite3.connect(config.DB_PATH)
    pool = pd.read_sql("SELECT * FROM winback_pool", con)
    aov = pool["first_order_value"].mean()
    con.close()
    n = len(pool)
    print(f"目标池 {n:,} 人（AOV {aov:.1f} BRL）| 分流 seed={SEED_ASSIGN}")

    # ---------- 随机分流（确定性，可复现） ----------
    rng_a = np.random.default_rng(SEED_ASSIGN)
    treat = rng_a.random(n) < 0.5
    pool["group"] = np.where(treat, "treatment", "control")

    # ---------- 敏感性演练：多档注入效应 ----------
    rows = []
    sim_results = {}
    for uplift in UPLIFT_GRID:
        rng_o = np.random.default_rng(SEED_OUTCOME + int(uplift * 100))
        p_base = pool["prob_cal"].values
        p_t = p_base[treat] * (1 + uplift)
        p_c = p_base[~treat]
        out = np.zeros(n)
        out[treat] = rng_o.random(int(treat.sum())) < p_t
        out[~treat] = rng_o.random(int((~treat).sum())) < p_c
        pool[f"y_{uplift}"] = out

        n_t = int(treat.sum())
        n_c = n - n_t
        res = run_analysis(out[treat].mean(), n_t, out[~treat].mean(), n_c)
        # ROI（处理组实际复购数 × 券面；增量按观测差 × 总人数）
        cost = out[treat].sum() * COUPON
        incr_profit = res["diff"] * n * aov * MARGIN
        roi = (incr_profit - cost) / cost if cost else float("nan")
        rows.append(dict(注入效应=f"+{uplift*100:.0f}%" if uplift else "0%（负对照）",
                         处理组率=f"{res['rate_t']*100:.2f}%",
                         对照组率=f"{res['rate_c']*100:.2f}%",
                         绝对Δ=f"{res['diff']*100:+.2f}pp",
                         相对=f"{res['rel']*100:+.1f}%" if not np.isnan(res['rel']) else "-",
                         p值=f"{res['p']:.2e}",
                         判定=res["sig"], ROI=f"{roi:+.2f}x"))
        sim_results[uplift] = res
        print(f"  注入 +{uplift*100:3.0f}%: Δ={res['diff']*100:+.2f}pp, "
              f"p={res['p']:.1e} → {res['sig']}, ROI {roi:+.2f}x")
    tab = pd.DataFrame(rows)

    # ---------- 主情景（+20%）的分组可视化 ----------
    u_main = 0.20
    res = sim_results[u_main]
    fig = go.Figure()
    fig.add_bar(x=["对照组", "处理组（发 R$5 券）"],
                y=[res["rate_c"] * 100, res["rate_t"] * 100],
                text=[f"{res['rate_c']*100:.2f}%", f"{res['rate_t']*100:.2f}%"],
                textposition="outside")
    fig.update_layout(yaxis_title="180 天复购率 %", xaxis_title="",
                      title="SIMULATED：+20% 注入效应演练（非真实实验）")
    lo, hi = res["ci"]
    fig.add_annotation(x=1, y=res["rate_t"] * 100, showarrow=False,
                       text=(f"Δ={res['diff']*100:+.2f}pp "
                             f"(95% CI {lo*100:+.2f} ~ {hi*100:+.2f}pp, p={res['p']:.2e})"))

    # ---------- 策略矩阵：品类 × 价值层 ----------
    seg = (pool.groupby(["first_category", "high_value"])
               .agg(p0=("prob_cal", "mean"), n=("prob_cal", "size"))
               .query("n >= 500").reset_index())
    seg["rel_lift"] = seg["p0"] * (1 + u_main) - seg["p0"]
    seg["incr_per_seg"] = seg["rel_lift"] * seg["n"]
    seg["cost"] = (seg["p0"] * (1 + u_main)) * seg["n"] / 2 * COUPON  # 处理组半量
    seg["profit"] = seg["incr_per_seg"] * aov * MARGIN
    seg["roi"] = (seg["profit"] - seg["cost"]) / seg["cost"]
    seg["first_category"] = seg["first_category"].astype(str).str.slice(0, 22)
    seg["high_value"] = np.where(seg["high_value"], "高价值", "普通")
    seg = seg.sort_values("incr_per_seg", ascending=False)
    matrix_md = seg.to_html(index=False, border=0, justify="left")
    print("\n策略矩阵（n≥500 的品类×价值层，按增量复购排序，+20% 情景）:")
    print(seg.head(12).to_string(index=False))

    # ---------- 输出 ----------
    tab_html = (tab.to_html(index=False, border=0)
                + "<br><small>注：「相对」为观测值 (处理组率−对照组率)/对照组率；注入效应定义为"
                  "每个客户自身基线概率上的 +x% 相对提升。基线率仅 ~3%、单臂二项抽样 "
                  "SE≈0.09pp，故观测相对值与注入值存在抽样偏差，属预期现象。</small>")
    blocks = [
        ("SIMULATED：分组复购率（注入 +20%）", fig),
        ("敏感性演练：流水线能否检出各档注入效应（含 0% 负对照）", tab_html),
        (f"策略矩阵：品类 × 价值层（+20% 情景，ROI 按 R${COUPON:.0f} 券）",
         matrix_md + "<br><small>注：增量/成本均为 SIMULATED 假设下的量级估算；"
                    "上线前需用真实 A/B 校准 uplift。</small>"),
    ]
    save_report(blocks, config.OUTPUTS_DIR / "ab_simulation.html",
                "挽回券 A/B 离线模拟演练（SIMULATED）")
    print(f"\nwrote {config.OUTPUTS_DIR / 'ab_simulation.html'}")
    print("\n诚实声明：本模块所有「实验结果」均为注入假设效应后的离线模拟，"
          "用途是验证分析流水线与展示设计，不代表真实因果效应。")


if __name__ == "__main__":
    main()
