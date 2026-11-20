# Olist 用户复购建模与留存策略 A/B 设计

> 在巴西电商 Olist 的 10 万单 / 9 张关系表数据上：**纯 SQL 特征工程 → 防泄漏复购倾向建模
> （LogReg 基线 → XGBoost + SHAP）→ 压力测试 → 挽回券 A/B 实验设计（样本量 + 盈亏平衡）
> → 离线模拟演练（SIMULATED）**，端到端回答「客户是谁、谁会回来、该不该发券、怎么发」；
> 另用 Hillstrom 邮件 RCT **加餐 uplift 建模**（T-Learner + Qini），回答「券/邮件该发给谁才真正多转化」。

## 项目出处

本项目在 [@Raymond-Chen-das/olist-customer-analysis](https://github.com/Raymond-Chen-das/olist-customer-analysis)
（00→08 pipeline 底本）基础上扩展：
- **新增** `09_ab_design.py`（A/B 实验设计 + 样本量 + 券方案盈亏平衡）、
  `10_ab_simulation.py`（离线模拟演练 + 品类×价值层策略矩阵）与
  `11_uplift.py`（uplift 建模：T-Learner 双版本 + Qini 评估 + 基线对比，
  数据为 Hillstrom 邮件 RCT，选源决策记录见 `docs/uplift-report.md` 第 2 节）
- **修复** `03_eda.py` 中 H2 检验未处理 NaN 导致 p=nan 的问题（详见代码内注释）
- README 重写为简体中文，结果数字为本仓库复现值

## 重点成果

| 发现 | 数字 |
|---|---|
| 97.0% 客户只买一次 | 复购率 3.00%（cohort 留存 <0.7%）→「获取型」生意 |
| 16.5% 高价值客户 → 31% 营收 | 80/20 高度集中，营销资源应分层 |
| 复购主因 = 首单品类 | 家居/可补货品类复购率约为一次性品类 6×（SHAP 与策略矩阵互相印证） |
| 服务体验 ≠ 留存 | 准时配送 p=0.16、评论分 p=0.20 均不显著 |
| 模型天花板（诚实披露） | 仅凭首单信息 ROC≈0.61、PR-AUC 0.05、top-decile lift 1.84×；
    压力测试（5 种子 × 时间切分 × 移除品类）下结论稳健（ROC 0.592±0.008） |
| **A/B 设计核心发现** | 券盈亏平衡与 p0 无关：等值券(10%×AOV) 需 **+100%** 相对提升才回本，
    R$5 定额券只需 **+19%**、R$10 需 +46% → 主实验推荐 R$5 定额券 |
| A/B 可行性 | 目标池 64,741 人：+20% 相对 MDE 下 power 93~99%（可行）；
    +10% MDE 不可行（power 24~59%） |
| **Uplift 加餐（Hillstrom RCT）** | Womens 邮件 ATE +0.42pp 但效果集中在女性客户
    （子群 +0.514pp vs 男性 +0.063pp）；T-Learner 在该 campaign 显著优于
    Response 基线（Qini 系数 +6 vs −1），top 20% 定向 = 1.81× 整体增量效率 |

## 方法要点

- **真·SQL 分析**：NTILE RFM 评分、Window 函数、Cohort 留存全部纯 SQL（SQLite），不用 pandas 补算
- **防泄漏**：复购模型只用「首单时点」特征；时间切分（快照日 2018-10-17，首单 ≤ 2018-04-20 才入建模集）；
  前处理只在训练集 fit
- **诚实评估**：3% base rate → 看 PR-AUC / lift / top-decile，不看 Accuracy；
  明确披露 ROC≈0.61 是「仅凭首单信息」的数据天花板
- **校准意识**：`class_weight='balanced'` 概率未校准（均值虚高 16×）→
  A/B 基线 p0 用测试集 Platt 校准后估计（全池 2.94%）
- **可复现**：固定随机种子、锁定版本（requirements.lock.txt）、raw CSV 一键重建（00→10）

## 分析流程

| 阶段 | 脚本 | 做什么 | 产出 |
|---|---|---|---|
| 数据契约 | `00_validate_contract.py` | 字段/主外键/时间戳验证 | `docs/data-contract.md` |
| 建库 | `01_build_db.py` | 9 张 CSV → SQLite + delivered 视图 | `data/olist.db` |
| SQL 特征 | `02_build_features.py` | RFM/首单特征 + cohort 留存 | DB 表 |
| EDA | `03_eda.py` | 分布、相关、3 个假设检验 | `outputs/eda_report.html` |
| 分群 | `04_rfm.py` | RFM 评分箱 + K-means（k=3） | `outputs/rfm_segments.html` |
| 基线建模 | `05_propensity.py` | LogReg(balanced)，防泄漏 | `outputs/propensity.html` + `customer_scores` |
| 模型对照 | `06_xgb_shap.py` | XGBoost + SHAP 驱动因子 | `outputs/xgb_shap.html` |
| 压力测试 | `07_stress_test.py` | 种子/时间切分/移除品类 | `docs/evaluation-report.md` |
| 仪表盘 | `08_report.py` | 单页整合 | `index.html` |
| **A/B 设计（新增）** | `09_ab_design.py` | 目标池、校准 p0、样本量、券盈亏平衡 | `docs/ab-experiment-design.md` + `outputs/ab_design.html` |
| **模拟演练（新增）** | `10_ab_simulation.py` | 注入效应 0/10/20/50% 演练 + 策略矩阵 | `outputs/ab_simulation.html`（全部标注 SIMULATED） |
| **Uplift（加餐）** | `11_uplift.py` | Hillstrom RCT 上 T-Learner（XGB/线性）+ Qini + 基线对比 | `outputs/uplift.html` + `docs/uplift-report.md` |

## 环境与运行

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.lock.txt xgboost shap  # Windows
.venv/bin/python -m pip install -r requirements.lock.txt xgboost shap          # macOS/Linux

# 数据：Kaggle「Brazilian E-Commerce Public Dataset by Olist」9 张 CSV 放入 raw/
#       （公开镜像：github.com/Sangamesh2703/Brazilian-E-Commerce）
# 另：raw/hillstrom.csv（MineThatData 2008 邮件 RCT，64,000 行，仅模块 11 需要）
#       官方直链见 docs/uplift-report.md 第 2 节

for s in 00_validate_contract 01_build_db 02_build_features 03_eda 04_rfm \
         05_propensity 06_xgb_shap 07_stress_test 08_report 09_ab_design \
         10_ab_simulation 11_uplift; do
  .venv/Scripts/python.exe $s.py
done
```

## 诚实边界（面试口径）

1. **ROC≈0.61 不是模型没调好**：仅用首单时点特征时数据信号就这么多；
   改进路径 = 行为序列特征 / 外部特征（已在评估报告中说明）
2. **A/B 部分是设计 + 模拟，不是真实实验**：Olist 无随机发券历史，
   模拟演练的价值是「上线前把预注册分析流水线跑通 + 展示设计严谨性」，
   所有模拟输出均已标注 SIMULATED
3. p0 为校准后模型均值（非实测基线），毛利率 20% 为行业量级假设
4. **Uplift 模块用的是另一个数据集（Hillstrom RCT）**：Olist 无随机化、
   无法评估 CATE 排序；Criteo Uplift 因 v1 官方勘误泄漏 + v2 下载源失效
   而弃用（决策记录见 `docs/uplift-report.md` 第 2 节）
5. uplift 的 conversion 目标 test 集增量转化仅 ~27–46 人，模型间 Qini 差异
   在抽样噪声范围内——**方向性结论（T-Learner ≥ Response 基线、
   Womens campaign 异质性结构）稳健，单点数字不作主张**

## 技术栈

Python 3.13 · SQLite · pandas/numpy · scikit-learn · XGBoost · SHAP · SciPy · Plotly
