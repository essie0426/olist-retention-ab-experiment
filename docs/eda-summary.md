# EDA 摘要（阶段二）

- 数据：`customer_features`（93,358 位客户，delivered 订单）；建模集 66,717 位（首单 ≤ 2018-04-20）。
- 可视化：`outputs/eda_report.html`。
- 原则：假设于规划阶段已定，本阶段为验证；负面结果如实呈现。

## 1. 分布与偏态 → 方法含义
| 特征 | 观察 | 含义 |
|---|---|---|
| monetary / avg_order_value / first_order_value | 极右偏 skew≈9（9.6 → 13,664） | 分群前对金额取 log |
| frequency | **97% = 1 单**，skew=11 | **F 维度近乎无变异 → 传统 RFM 对 Olist 实质退化为「R+M」**；3% 复购客视为特殊小群另处理 |
| recency_days | mean 285、48–761、skew 0.45 | 近似对称，可直接用 |
| first_review_score | 左偏，mean 4.14、众数 5 | 多数人给高分 |
| first_delivery_days | 右偏 mean 12、max 209 长尾 | 有离群，建模需 winsorize/log |

## 2. 目标变量
- `repurchase_180d` 正样本率 **2.96%**（1,976 / 66,717），严重不平衡。
- → 评估用 **PR-AUC / lift / top-decile 捕获率**，不用 Accuracy；建模用 `class_weight`。

## 3. 特征 → 复购关系（含诚实负面结果）
| 假设 | 检验 | 结果 | 判定 |
|---|---|---|---|
| H1 首单准时 → 复购 | chi-square | OR=1.13, phi=0.005, p=0.16 | **不显著（基本无关）** |
| H2 首单评论分 → 复购 | Mann-Whitney U | p=0.20（剔除 615 个无评论客户后复算），rank-biserial=−0.014，中位数皆 5 | **不显著（基本无关）** |
| H3 配送天数 → 评论分 | Spearman | rho=−0.237, p≈0, n=93,350 | **显著、效果中等** |

**结论：首单「服务体验」（配送、评论）不能预测复购；但配送质量确实影响满意度。**
→ 服务质量是「满意度杠杆」，不是「留存杠杆」。这个负面结果本身就是洞察。

## 4. 真正带复购信号的特征（补测）
| 特征 | 信号 | 用法 |
|---|---|---|
| **首单品类** | 复购率 1.4%–9.0%（baseline 2.96%，~6×）。高：home_appliances 9.0%、fashion_bags 5.1%、bed_bath_table 4.7%(n=6368)、furniture_decor 4.4%；低：cool_stuff 1.4%、electronics 1.7%、consoles 1.9% | **模块四主信号**（家用/可补货 vs 一次性耐久/嗜好） |
| 客户所在州 | 1.7%（CE/MA）– 3.7%（MT）/ 3.2%（SP, n=26,872） | 中等地理信号，纳入 |
| 首单付款方式 | credit/boleto 2.95%，几乎无差异 | 不纳入 |

## 5. 相关性 / 共线性
- monetary ~ avg_order_value ~ first_order_value：Spearman ρ≈0.98–0.99（因 97% 单笔购买三者趋同）。
- → 模块三 M 只取 `monetary`；模块四只取 `first_order_value`（且为非泄漏特征）。

## 6. 方法决定（EDA → 模块三/四）
- **模块三 RFM**：以 R、log(M) 为主轴，F 退化情况记录在案；K-means 前标准化；k 由轮廓系数＋业务可解释性决定。
- **模块四 复购挖掘**：特征 = 首单品类（分组/编码）＋ 客户州 ＋ first_order_value ＋ 首单体验（预期贡献低，诚实保留作对照）；LogReg(`class_weight`) 为主、XGBoost 对照；评估 PR-AUC / lift。
- **数据切分**：分层抽样（维持 2.96%）固定种子为主；时间切分（首单早→晚）作为阶段四稳健性检查。

## 7. 待处理
- 首单 73 个品类多数稀疏 → 需分组（领域分组／或 top-N one-hot + other）；编码须避免用目标值泄漏。
