# 实验记录（阶段三）

## 模块三：RFM 分群

### 传统 RFM 评分箱（SQL `NTILE`）
- R、M 各 `NTILE(5)`；F 因退化（97% frequency=1）改用实际次数规则评分。
- 6 个命名客群（分布 15–20%）：

| 客群 | 占比 | 经营建议（初版，模块五 A/B 设计细化） |
|---|---|---|
| 高价值 Champions | 16.5% | VIP 维系、新品优先、推荐计划 |
| 忠诚潜力 Loyal | 20.0% | 升级诱因、交叉销售 |
| 高额流失风险 At Risk | 15.5% | 召回优惠、满意度关怀 |
| 新客待培养 New | 15.4% | 第二单诱因、首单体验保障 |
| 一般 Regular | 16.0% | 轻度经营、观察 |
| 沉睡 Hibernating | 16.6% | 低成本唤醒或放生 |

### K-means 对照（R + log(M)，F 退化故排除）
实验：k=2..8 轮廓系数（StandardScaler，random_state=42，sample=10000）

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| silhouette | 0.341 | **0.359** | 0.343 | 0.332 | 0.351 | 0.337 | 0.343 |

- **选 k=3**（[3,6] 商业可解释区间内轮廓最高；避免 k=2 的琐碎二分）。
- 诚实注记：轮廓 ~0.36 偏低 → 客户价值是连续谱、非天然分群。
- k=3 群轮廓：
  - 群1（近期·高额：recency 222、M 328）→ 对应 Champions/Loyal
  - 群2（近期·小额：recency 194、M 71）→ 对应 New/Regular
  - 群0（久未·中低额：recency 472、M 120）→ 对应 Hibernating/At Risk
- 交叉比对：Champions 99% 落群1、New 100% 落群2、Hibernating 主要落群0。

### 方法选择结论
**以 RFM 评分箱客群为行动依据**（更细致、可直接解读、对应经营动作）；**K-means 作为结构验证**（独立还原同样 R×M 结构，证明分群非武断切点的产物）。

---

## 模块四：复购倾向 / 高潜力客户挖掘

- 问题框架：排序问题（在 ~3% 正样本中捞高潜力少数），非分类。
- 特征：首单品类 + 客户州 + log(first_order_value) + 首单体验（皆首单当下已知，无泄漏）。
- 模型：LogisticRegression(class_weight='balanced')，分层 75/25 切分，seed=42。

### 特征与实验总表
特征（皆首单当下已知，无泄漏）：首单品类、客户州、log(first_order_value)、首单体验（评论/配送/准时）、首单品项数、运费比、购买月份。分层 75/25 切分，seed=42。

| # | 模型 / 特征 | PR-AUC (×base) | ROC-AUC | Top-decile lift |
|---|---|---|---|---|
| 1 | LogReg(balanced)，基础首单特征 | 0.0475 (1.60×) | 0.592 | 1.64× |
| 2 | LogReg(balanced)，+SQL 扩展(品项/运费比/月份) | 0.0501 (1.69×) | **0.611** | 1.82× |
| 3 | XGBoost(scale_pos_weight)，同特征 | **0.0611 (2.06×)** | 0.603 | **1.86×** |

→ 三者落在同一天花板（ROC ~0.60–0.61、lift ~1.8×）。**模型选择不是瓶颈，数据才是。**

### 驱动因子
- **LogReg 胜算比（线性）**：home_appliances 3.6、home_confort 2.2、fashion_bags/bed_bath/furniture ≈1.7–1.8 推升；luggage 0.49、books 0.57、electronics/cool_stuff/consoles ≈0.6 压低 → 家用/可补货 vs 一次性耐久/嗜好。
- **XGBoost SHAP（非线性）**：主贡献 first_freight_ratio、log(first_order_value)、bed_bath_table、购买月份(3–4月)、配送天数。树模型多抓连续特征的非线性，线性模型靠品类，两者殊途同归。

### 诚实评估
绝对值偏弱（ROC ~0.6、lift ~1.8×），是 EDA 已预示的天花板：仅凭首单交易资料，复购接近不可预测。项目价值在 (1) 排序框架与不平衡指标、(2) 零泄漏、(3) 可解释驱动因子转成可行动洞察（品类＝获取/交叉销售杠杆）。要真正提升，需 Olist 没有的行为/营销资料。

### 交付选择
**LogReg(B) 为主交付模型**（可解释、胜算比直接对应动作）；XGBoost+SHAP 作对照，佐证天花板由数据决定。

---

## 模块七：Uplift 建模（11_uplift.py，2026-11-20 完成）

- 问题框架：定向排序从 p(Y|x)（saver trap）切换为 CATE = P(Y=1|x,T=1) − P(Y=1|x,T=0)。
- 数据源决策（详见 docs/uplift-report.md §2）：Criteo v1 有官方勘误泄漏、v2.1 官方链接 404 且
  托管方 HF 不可达 → 改用 Hillstrom/Meggitt 2008 邮件 RCT（64,000 行，segment 随机化，
  三组基线均衡已核验）。mens/womens 列经交叉验证判定为顾客性别属性而非投放标志。
- 任务：Mens / Womens 两个 campaign，共用 No E-Mail 对照组；主目标 conversion、
  辅助目标 visit；按 segment 分层 70/30，seed=45。
- 方法：T-Learner 双版本（XGBoost 保守配置 150 树/depth3/mcw100——初版 300 树/depth4
  实测过拟合：臂内正样本仅 87–186 个，CATE 退化为噪声之差；线性 LogReg C=0.5 稳健性对照）；
  基线：随机 / Response 池化 / Treatment-only。指标：ATE、Qini 曲线/系数、top 20% 切片。
- 结果：Mens ATE +0.720pp、Womens ATE +0.419pp；Womens 上 T-Learner(线性) Qini 系数 +6
  显著优于 Response −1（top20% 增量 = 1.81× 整体）；Mens 上各模型 Qini 差异 ±5 在噪声范围内
  （test 增量转化仅 ~27–46 人）。visit（稠密目标）上 Womens：T-Learner +47/+50 vs Response +2。
- 统计口径：只主张方向性结论；conversion 目标单点 Qini 数字不作推断（事件数不足）。
- 交付：outputs/uplift.html + docs/uplift-report.md。
