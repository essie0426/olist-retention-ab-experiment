# 模块 11 · Uplift 建模：识别「可被说服」的客户

> 状态：已完成 | 脚本：`11_uplift.py` | 展示页：`outputs/uplift.html` | 数据：`raw/hillstrom.csv`

## 1. 背景与动机：为什么 09/10 之后还需要 uplift

09/10 的挽回池按**复购概率** p(Y|x) 排序选对象。uplift 文献把这称为 **saver trap**
（"省钱陷阱"）：p(Y|x) 最高的一批人往往是 **sure thing**——不发券不发邮件也会买，
对他们定向的边际收益接近 0，钱花在了本就会发生的事上。

uplift 建模把定向排序从「谁最可能转化」切换为「谁最可能被干预说服」，直接估计
个体处理效应：

```
CATE(x) = P(Y=1 | x, T=1) − P(Y=1 | x, T=0)
```

只有 RCT（随机化实验）数据能无偏地评估 CATE 排序。Olist 是历史交易数据、
没有随机化，所以本模块**换一个带 RCT 的数据集**来完整走一遍 uplift 流水线——
这与 10 号模块（在 Olist 上做模拟演练）是同一思路的两半：一个练"设计"，一个练"评估"。

## 2. 数据源决策记录（诚实说明）

**原计划：Criteo Uplift Prediction v2.1**（2018 年 Criteo 官方 uplift 竞赛数据集）。
排查结论：

| 版本 | 问题 | 结论 |
|---|---|---|
| v1（25.3M 行） | **官方勘误（erratum）**：各广告商真实 lift 水平不均，模型可以用「广告商身份」特征人为抬高 uplift 指标，排名失真 | 不可用（教学/作品场景下结论不可信） |
| v2.1（无偏重采样版，13,979,592 行 / 297MB gz） | 官方页 `ailab.criteo.com` 的下载链接 `go.criteo.net/criteo-research-uplift-v2.1.csv.gz` 已 404；正式托管方 HuggingFace（`datasets/criteo/criteo-uplift`）在当前网络环境不可达 | 不可达 |

**改用：MineThatData Hillstrom/Meggitt 2008 邮件营销 RCT**（`raw/hillstrom.csv`）。
选择理由：

1. **经典 RCT**：2008 年真实 A/B 实验，`segment` 即随机化组（No E-Mail /
   Mens E-Mail / Womens E-Mail 各 ~21,333 人），随机化有效性可核验（见第 3 节）；
2. **数据源可追溯**：原始文件直接来自主办方 minethatdata.com（3.96MB，64,000 行），
   同时被 Zenodo 公开数据包（record 23312512，《Risk Is Not Uplift》论文配套）引用，
   其 `scripts/download_data.sh` 提供了同一官方直链，交叉验证一致；
3. **结构合适**：两组实验邮件（Mens/Womens）+ 一个共同对照组，天然构成两个
   uplift 任务；样本量 64k 在本环境下完全跑得动。

代价（如实承认）：Hillstrom 只有 12 个特征（无浏览/点击行为），异质性因子有限；
Criteo 的 12 维匿名数值特征 + 297MB 规模做出来"看起来"更唬人，但结论含金量更低。
**数据源故事本身就是这个模块的一部分**——面试时讲"为什么没用 Criteo"比讲结果更值钱。

## 3. 数据契约与任务定义

**契约核验**（`11_uplift.py` 启动时自动断言）：

- 64,000 行 × 12 列，0 缺失
- `segment` 三组均衡：Mens E-Mail 21,307 / No E-Mail 21,306 / Womens E-Mail 21,387
- **RCT 有效性**：基线特征三组均值几乎相同（recency 5.774/5.750/5.768，
  history $242.8/$240.9/$242.5）→ 随机化成立
- `zip_code` 只有 3 个取值（匿名化区域），`history_segment` 为 7 级历史消费带，
  `channel` 为 3 类（Web/Phone/Multichannel），均保留为类别特征

**关键语义判定**（防止 treatment 定义错误）：`mens`/`womens` 两列**不是投放标志**。
证据：对照组（No E-Mail）内 `mens=1` 仍占 55.3%——若它是投放组，对照组里应为 0%。
它是**顾客性别属性**。两个 campaign 都是发给全量人群的，命中性别偏好与否才是
异质性来源——这恰是模型要学的。

**任务**（共用同一对照组，主目标 conversion，辅助目标 visit）：

| campaign | treatment | control |
|---|---|---|
| Mens | `segment == "Mens E-Mail"` | `segment == "No E-Mail"` |
| Womens | `segment == "Womens E-Mail"` | `segment == "No E-Mail"` |

特征：recency, history, history_segment, zip_code, newbie, channel, mens, womens（8 个）。

## 4. 方法

**T-Learner（双版本）**：两臂各训练一个模型，CATE = p̂1(x) − p̂0(x)。

- **XGBoost 臂**：保守配置 150 树 / depth 3 / lr 0.03 / min_child_weight 100 /
  reg_lambda 5 / reg_alpha 1，`scale_pos_weight` 按臂内正率。
  为什么保守？conversion 臂内正样本只有 **87–186 个**（~0.5–1.3%），初版用
  300 树 / depth 4 / mcw 5 时两臂模型各自记住噪声，CATE 退化为"两个噪声之差"，
  排序与随机无异（Qini 系数 ≈ 0，已实测）。强正则让臂模型只保留最强信号。
- **线性臂**（LogReg C=0.5 + 类别 one-hot）：稳健性对照——低事件数下线性模型
  方差更小，同时检验"非线性到底有没有带来增益"。

**基线**：

| 基线 | 逻辑 | 预期失败模式 |
|---|---|---|
| random | 随机定向 | 参照线，Qini 系数 = 0 |
| response 单学习器 | 全量池化数据预测 P(Y=1|x)，无处理信息 | saver trap：选中 sure thing |
| treatment-only 单学习器 | 只用处理臂训练 P(Y=1|x,T=1) | 找不到"对谁无效"，无法扣掉自然转化 |

**切分与评估**：按 segment 分层 70/30，seed=45（与 09/10 的 42/43/44 独立）；
同一 campaign 的 conversion 与 visit 共用同一份切分。test 集本身是 RCT，
因果指标直接可用。

**指标**：

- **ATE**：test 集两比例差（总体平均效应）
- **Qini 曲线**（Radcliffe 2007）：按预测分降序，
  `Qini(k) = 累计处理组转化 − 累计对照组转化 × nT(k)/nC(k)`，单位：人
- **Qini 系数** = 模型曲线面积 − 随机线面积（单位：增量转化数；随机定向恒为 0）
- **top 20% 切片**：模型选出的前 20% 人里，处理组 vs 对照组转化率之差
  （业务语言：如果只能发给 20% 的人，人均增量是多少）

## 5. 结果

### 5.1 总体效应（test 集）

| campaign | 目标 | ATE | 相对提升 | test 内增量转化总数 |
|---|---|---|---|---|
| Mens | conversion | +0.720 pp（1.267% vs 0.548%） | +131% | ≈ 46 人 |
| Mens | visit | +8.03 pp（18.3% vs 10.3%） | +78% | ≈ 514 人 |
| Womens | conversion | +0.419 pp（0.904% vs 0.485%） | +86% | ≈ 27 人 |
| Womens | visit | +5.04 pp（15.3% vs 10.3%） | +49% | ≈ 323 人 |

### 5.2 Qini 系数（模型 − 随机，单位：增量转化数；test 集）

**Mens campaign · conversion**（test 增量总数 ≈ 46 人，噪声水平高）

| 模型 | Qini 系数 | top20% 增量 | top20% 对照组率* |
|---|---|---|---|
| T-Learner (XGBoost) | −3 | 0.557 pp | 0.47% |
| T-Learner (线性) | −2 | 0.678 pp | 0.87% |
| Response 单学习器（池化） | 0 | 1.015 pp | 1.10% |
| Treatment-only 单学习器 | −1 | 0.852 pp | 1.33% |

**Womens campaign · conversion**（T-Learner 明显占优）

| 模型 | Qini 系数 | top20% 增量 | top20% 对照组率* |
|---|---|---|---|
| **T-Learner (线性)** | **+6** | **0.757 pp（1.81× 整体 ATE）** | 0.69% |
| T-Learner (XGBoost) | +3 | 0.652 pp | 0.38% |
| Treatment-only 单学习器 | 0 | 0.328 pp | 1.01% |
| Response 单学习器（池化） | −1 | 0.655 pp | 0.88% |

**visit（辅助目标，更稠密 → 检出力演示）**

| campaign | XGB-TL | 线性 TL | Response | Treat-only |
|---|---|---|---|---|
| Mens | −1 | +21 | +23 | +25 |
| **Womens** | **+47** | **+50** | **+2** | +18 |

\* top20% 切片内对照组转化率越高 = 该模型越倾向选中 sure thing（saver trap 的直接证据）。
注意 Mens 一栏：Treatment-only 选出的 top 20% 里对照组也有 1.33% 的人"不发也会买"，
而 T-Learner (XGBoost) 同切片的对照组率只有 0.47%——前者在发"本来就会买的单"，
后者在发"被邮件说服的单"。

### 5.3 Ground truth 核验（全量子群 ATE，未参与训练/评估）

| 子群 | 子群 ATE | 对比 |
|---|---|---|
| Womens 邮件 → womens=1 客户 | **+0.514 pp** | womens=0 仅 +0.063 pp（差 8 倍） |
| Mens 邮件 → mens=1 客户 | +0.769 pp | mens=0 +0.572 pp（差 1.3 倍） |
| recency ≤ 3（Mens） | +0.733 pp | 其余 +0.651 pp（弱） |

**异质性真实存在且结构清晰**：Womens 邮件的效果几乎全部集中在女性客户身上；
Mens 邮件对两种性别都有效。一个规则「Womens 邮件只发给 womens=1 客户」
（55% 人群）就能拿到 +0.514 pp 的子群 ATE，高于 campaign 整体 +0.419 pp——
这不需要任何模型，但模型的价值在于同时处理多个因子并推广到新特征。

## 6. 发现与解读

1. **T-Learner 在「处理效应强异质」的场景赢得干净**：Womens campaign 上，
   T-Learner（两版本）的 Qini 系数（+6/+3，conversion；+50/+47，visit）
   系统性高于 Response 基线（−1/+2）——这正是 uplift ≠ response 的教科书案例：
   池化模型分不清"谁因为邮件才买"，而 Womens 邮件对约 45% 的男性客户
   几乎无效（+0.063 pp），Response 模型却把他们当高响应人群。
2. **Mens campaign 上模型间差异在噪声范围内**（Qini 系数 ±5，而 test 增量
   总数 ≈ 46）：真实 CATE 结构弱（两种性别都有效），没有"该排除谁"的强信号，
   T-Learner 的减法优势无处发挥。**这本身就是重要发现**：不是所有 campaign
   都值得上 uplift 模型——先做子群 ATE 探查，效应结构弱的直接全量发。
3. **样本量与事件数决定上限**：conversion ~1% 意味着 test 集增量转化只有
   ~27–46 人，任何排序指标的抽样误差都很大；换到更稠密的 visit 后，
   同一套流水线的区分力立刻出来（Womens：+50 vs +2）。**业务建议：
   低转化率的 uplift 实验，要么拉长实验窗口攒事件，要么用前置指标
   （visit/加购）做中期读数、主指标做最终裁决。**
4. **saver trap 被直接量化**：top20% 切片的对照组率对比（5.2 节）显示
   Treatment-only 基线选出的"高响应人群"里有大量不发也会买的人
   （Mens top20% ctrl 率 1.33% vs T-Learner XGB 0.47%）。

## 7. 局限与后续

- **T-Learner 方差偏大**：两臂独立建模，臂内样本少时 p̂1、p̂0 各自噪声大；
  X-learner / R-learner / DR-learner 在小样本下通常更稳，是自然的下一步
  （X-learner 在极端不平衡的组间样本量上还有额外优势，本数据组间均衡，收益有限）。
- **CATE 未校准**：低正率下臂模型概率有偏，本模块只评估排序，不做概率解释；
  若要做"券面额个性化"需要校准后的 CATE。
- **单一时间窗、二元结果**：没有 spend/复购金额目标，没有多波次邮件序列
  （真实场景里第 2 封邮件的 uplift 依赖第 1 封是否送达/被打开）。
- **未做多重检验校正**：两个 campaign × 两个目标共 4 组评估，若做严格推断
  需 Bonferroni / FDR 校正；本模块定位是方法论流水线验证，不做 p 值主张。
- **Hillstrom 特征只有 12 列**：真实 uplift 系统的胜负手通常在行为特征
  （打开/点击历史、渠道偏好），本数据无法体现这一层。

## 8. 复现

```bash
# 数据：raw/hillstrom.csv（见 README 数据源说明）
./.venv/Scripts/python.exe 11_uplift.py
# → outputs/uplift.html（Qini 曲线 / 十分位切片 / 基线对比）
# 全部随机源固定：SEED=45（切分/绘图随机数/模型 random_state 同源）
```
