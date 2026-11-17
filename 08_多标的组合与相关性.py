# -*- coding: utf-8 -*-
# 来源：Quant-for-babies / ThreshoLdnnis
# 原始 Notebook：notebooks/phase2_intro/04_多标的组合与相关性.ipynb
# 来源链接：https://github.com/ThreshoLdnnis/Quant-for-babies/blob/711ca83a82f6f1bb1745c6ca28c1ed3ecd2a97bd/notebooks/phase2_intro/04_多标的组合与相关性.ipynb
# Git blob SHA：5b25102c784981898456137156f217222baac29a
# 导出说明：代码单元及原有注释原样保留；Markdown 单元转换为注释。
# 使用方式：按 # %% 分隔的单元顺序运行；环境及第 3 章数据源选择见 README.md。

# %% [markdown] Notebook cell 1 (markdown)
# # 《零基础学习量化金融》
# ## 从Python到AI量化交易实战 第二期
# ### 第8章：多标的组合与相关性

# %% [markdown] Notebook cell 2 (markdown)
# ---
# ## 本章你将学会
# 
# - ✅ 一次下载 **多只股票 + 大盘 ETF**，对齐成一张「宽表」
# - ✅ 计算 **相关性矩阵（Correlation Matrix）**，用数字描述「同涨同跌」
# - ✅ 画 **热力图**，一眼看出谁和谁「绑在一起」
# - ✅ 构造 **等权组合**，亲手验证：组合波动能否低于单票
# - ✅ 把 **分散投资（Diversification）** 从口号变成可复现的代码实验
# 
# **当前等级**
# 🎮 **Lv.5 组合分析师**
# 
# **本章难度**
# ⭐⭐⭐☆☆
# 
# **预计学习时间**
# 35～50 分钟（需联网拉取 AkShare 数据）
# 
# **前置知识**
# 
# - 完成 **第二～七章**（波动率、Sharpe、Beta、回撤与仓位）
# - 第一期 **第二章**（多标的 `pct_change`、宽表思维）
# 
# ---
# 
# > 💡 **一句话预告**：如果你只买 AAPL，你的命运和苹果绑死；如果你同时买苹果、微软、银行、能源——它们不会完全同步涨跌，组合的「心跳」往往比单票更平稳。
# 
# 「不要把鸡蛋放在同一个篮子里」——这句话人人都听过，但很少人用 **数据** 回答三个追问：
# 
# 1. **到底和谁同涨同跌？** 苹果和微软，真的像双胞胎吗？
# 2. **分散之后，波动真的变小了吗？** 还是只是心理安慰？
# 3. **我的组合，比大盘 SPY 更稳还是更野？**
# 
# 这一章，我们用 **5 只真实标的、约 3 年美股日线**，把这三个问题跑通。你会看到：当几只股票 **不完全相关** 时，等权组合的波动往往 **低于** 任意单票——这不是玄学，是 **协方差矩阵** 送给业余投资者的礼物。
# 
# 这也是 **多标的分析** 的起点：专业量化团队每天在做的事，本质上就是——**选对标的、看清关系、控制组合风险**。今天我们先走最朴素的一步：**等权 + 相关性**。

# %% [markdown] Notebook cell 3 (markdown)
# ### 环境准备
# 
# 如果提示缺少库，在项目根目录执行：`pip install -r requirements.txt`
# 
# > 本章行情改用 **AkShare**（`stock_us_daily`）拉取美股/ETF，无需 yfinance，大陆网络可直接使用。
# 
# 本格预先定义了 **`fetch_us_prices`、`equal_weight_returns`、`annualize_volatility`** 等工具函数——后面每一节都会复用，避免重复写下载与公式。跑通后应看到 `环境就绪 ✓`。

# %% Notebook cell 4 (code; source_chars=3280)
# ========== 环境准备：导入库 + 多标的 / 组合工具函数 ==========
import warnings                                   # 导入警告控制模块
warnings.filterwarnings('ignore')              # 隐藏次要警告，Notebook 输出更干净

import statistics as stats                      # 标准库：均值、中位数等
import numpy as np                              # 数值计算、矩阵运算
import pandas as pd                             # 表格数据处理
import matplotlib.pyplot as plt                 # 绘图
import akshare as ak                            # 国内金融数据接口（本章拉美股/ETF）
import time                                     # 批量请求间隔，避免触发接口限速

plt.rcParams['font.sans-serif'] = [              # 跨平台中文字体回退
    'PingFang SC', 'Microsoft YaHei', 'SimHei',
    'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans',
]
plt.rcParams['axes.unicode_minus'] = False      # 坐标轴负号正常显示

TRADING_DAYS = 252                              # 美股常用年化交易日数


def fetch_us_close(symbol: str, years: float = 3.0) -> pd.Series:  # 工具：拉取单只美股/ETF 收盘价
    """联网拉取单只美股/ETF 前复权收盘价。"""                      # 函数文档字符串
    start = (                                    # 计算样本起始日字符串
        pd.Timestamp.today()                     # 今天
        - pd.DateOffset(years=years, days=30)   # 往前 years 年，多留 30 天缓冲
    ).strftime('%Y-%m-%d')                       # 格式化为 'YYYY-MM-DD'
    df = ak.stock_us_daily(symbol=symbol, adjust='qfq')  # 发起 HTTP 请求 · 前复权
    if df is None or df.empty:                   # 网络/接口异常
        raise RuntimeError(f'{symbol} 未返回数据，请检查网络或 akshare 版本')  # 主动报错
    df['date'] = pd.to_datetime(df['date'])      # 字符串 → datetime
    close = (                                    # 整理为收盘价 Series
        df.set_index('date')                     # 日期列 → 行索引
          .sort_index()                          # 按时间升序
          ['close']                              # 只保留收盘价
          .loc[lambda s: s.index >= start]       # 截取 start 之后
          .rename(symbol)                        # 命名 = 股票代码
    )                                              # 链式调用结束，得到 close Series
    return close                                 # 返回单标的收盘价序列


def fetch_us_prices(tickers: list[str], years: float = 3.0) -> pd.DataFrame:  # 工具：批量下载宽表
    """批量下载并宽表对齐：列 = 标的，行 = 日期。"""                      # 函数文档字符串
    frames = {}
    for t in tickers:
        frames[t] = fetch_us_close(t, years)
        time.sleep(1)                            # 新浪接口限速，连续请求间隔 1 秒
    return pd.DataFrame(frames).dropna()         # 宽表对齐，删缺失行（非共同交易日）


def annualize_volatility(daily_returns: pd.Series) -> float:  # 工具：日收益 → 年化波动
    """日收益率 → 年化波动率（σ · √252）。"""                      # 函数文档字符串
    daily_std = daily_returns.std()             # 日收益标准差
    return daily_std * np.sqrt(TRADING_DAYS)    # 日波动 × √252


def equal_weight_returns(returns: pd.DataFrame, tickers: list[str]) -> pd.Series:  # 工具：等权组合日收益
    """等权组合日收益 = 各成分日收益的逐行平均。"""                  # 函数文档字符串
    subset = returns[tickers]                    # 取出组合成分列
    return subset.mean(axis=1)                   # axis=1：每行求平均 → 等权


def buy_hold_equity(daily_returns: pd.Series) -> pd.Series:  # 工具：买入持有净值曲线
    """买入持有净值 = 日收益连乘（起点 ≈ 1）。"""                  # 函数文档字符串
    growth = 1 + daily_returns                   # 日增长因子 1+r_t
    return growth.cumprod()                        # 连乘 → 净值曲线


print('环境就绪 ✓')                               # 提示：环境加载完成


# %% [markdown] Notebook cell 5 (markdown)
# ---
# ### 8.1 多标的：一张表对齐
# 
# 想象你同时盯着 5 块屏幕，每块播一只股票的 K 线——量化第一步，是把它们 **拼成一张 Excel 表**：
# 
# - **行** = 交易日（大家共用的日历）
# - **列** = 标的（AAPL、MSFT、JPM……）
# - **单元格** = 那天的收盘价或日收益率
# 
# 这样，`returns.corr()` 才能一次性算出「谁和谁更同步」。如果日期没对齐（有的股票缺数据），相关性就会算歪——所以我们会用 `dropna()` 只保留 **5 只都有报价** 的交易日。
# 
# 下面选 5 只 **行业差异明显** 的标的（可自行替换代码）：
# 
# | 代码 | 含义 | 为什么选它 |
# |------|------|-----------|
# | AAPL | 苹果 | 科技龙头，消费+硬件 |
# | MSFT | 微软 | 科技龙头，云+软件 |
# | JPM | 摩根大通 | 金融/银行，和利率、信贷周期更相关 |
# | XLE | 能源 ETF | 油价、通胀的「另一套剧本」 |
# | SPY | 标普500 ETF | 大盘基准，后面用来对比组合表现 |
# 
# > 🎬 **小剧场**：AAPL 和 MSFT 像同班同学的笔记——往往一起抄作业；XLE 像隔壁班的体育生——能源涨价时，它可能在开 party，而科技股在开会。

# %% Notebook cell 6 (code; source_chars=685)
# ========== 下载多标的并对齐（AkShare · 宽表）==========
TICKERS      = ['AAPL', 'MSFT', 'JPM', 'XLE', 'SPY']  # 4 只个股/ETF + 大盘 SPY
PERIOD_YEARS = 3                                       # 回溯约 3 年

print(f'akshare 版本: {ak.__version__}')                 # 打印库版本，便于排查
print('接口: ak.stock_us_daily(symbol, adjust="qfq")')   # 说明数据来源

prices  = fetch_us_prices(TICKERS, PERIOD_YEARS)       # 收盘价宽表：行=日期，列=标的
returns = prices.pct_change().dropna()                 # 简单日收益率 r_t = P_t/P_{t-1}−1

print(f'\n样本: {len(returns)} 个交易日 × {len(TICKERS)} 只标的')  # 样本规模
print(f'区间: {returns.index[0].date()} → {returns.index[-1].date()}')  # 起止日期
print(returns.head())                                # 预览前 5 行


# %% [markdown] Notebook cell 7 (markdown)
# #### 📊 读表：多标的宽表长什么样？
# 
# 运行上一格后，你会看到类似下面的信息：
# 
# - **样本规模**：约 700+ 个交易日 × 5 列——每行都是「5 只同时有报价」的日子。
# - **`returns.head()`**：第一行日收益通常很小（如 ±0.5%～2%），**不要**和 ρ 或年化波动混淆。
# 
# **三个读表习惯：**
# 
# 1. **看列名**：确认 5 只都在，没有拼写错误。
# 2. **看日期索引**：起止是否覆盖你想要的年份？太短则相关性不稳定。
# 3. **看缺失**：若行数突然变少，说明某标的在部分日期缺数据——`dropna()` 已帮你删掉不对齐的行。
# 
# > 这张表是本章的「原材料」。后面所有矩阵、热力图、组合，都是在这张 `returns` 上变魔术。

# %% [markdown] Notebook cell 8 (markdown)
# ---
# ### 8.2 相关性矩阵：量化「同涨同跌」
# 
# 两只股票今天都涨、明天都跌——这种 **步调一致性**，用 **Pearson 相关系数 ρ** 来度量：
# 
# | ρ 的范围 | 直觉 | 对分散投资意味着 |
# |---------|------|----------------|
# | **ρ ≈ 1** | 几乎同涨同跌 | 买两只 ≈ 买一只的放大版，分散效果差 |
# | **ρ ≈ 0** | 各走各的 | 组合可能更「稳」，是分散化的理想素材 |
# | **ρ ≈ -1** | 往往一涨一跌 | 对冲直觉（实际股票里很少长期维持） |
# 
# 矩阵对角线永远是 **1.0**（自己和自己的相关度）。我们关心的是 **非对角线**：比如 AAPL–MSFT 通常 **0.7～0.9**（高度同步），而 XLE 和科技股往往 **更低**。
# 
# `pandas` 一行 `returns.corr()` 就能算完整张表——下面运行后，先 **扫一眼数字**，再对照下一节的热力图颜色。

# %% Notebook cell 9 (code; source_chars=629)
# ========== 相关性矩阵：谁和谁「同涨同跌」？==========
corr = returns.corr()                                  # Pearson 相关系数 ρ，对称矩阵

print('日收益率相关系数（ρ ∈ [-1, 1]）：')              # 标题说明
print(corr.round(2))                                 # 保留两位小数，Notebook 美观显示

mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)   # 上三角掩码（k=1 排除对角线 1.0）
pairs = corr.where(mask).stack().sort_values()         # 展平为 MultiIndex Series 并升序
low_pair = pairs.index[0]                              # 最低相关的 (股票A, 股票B) 元组
low_rho  = pairs.iloc[0]                               # 对应的相关系数值
print(f'\n最低相关一对: {low_pair[0]} ↔ {low_pair[1]}，ρ = {low_rho:.2f}')  # 输出结论


# %% [markdown] Notebook cell 10 (markdown)
# #### 📊 读矩阵：数字告诉你的故事
# 
# 对照打印出的 **相关系数表**，建议按这个顺序扫：
# 
# 1. **对角线 1.00**：忽略，自己和自己永远完全相关。
# 2. **AAPL ↔ MSFT**：通常 **0.75+**，两只科技龙头「绑」得很紧——若组合里只有它俩，分散效果有限。
# 3. **XLE 与其他**：往往 **明显更低**——能源和科技的「剧本」不同，是分散化的主力队员。
# 4. **各股 ↔ SPY**：都在 **0.5～0.9** 一带很常见——个股很难完全脱离大盘。
# 
# 代码还会自动打印 **ρ 最低的一对**。想一想：若你专门挑「不太同步」的两只配在一起，组合波动理论上会更低——下一节用颜色验证这一点。

# %% [markdown] Notebook cell 11 (markdown)
# ---
# ### 8.3 热力图：一眼读懂关系
# 
# 数字矩阵准确，但 **人眼不擅长扫 5×5 的表格**。热力图把 ρ 映射成颜色：
# 
# - **深红** → ρ 接近 +1（强正相关，同涨同跌）
# - **白色/浅色** → ρ 接近 0（关系弱）
# - **深蓝** → ρ 接近 −1（负相关，少见）
# 
# 这是 **Phase 2 的收官成果图之一**——建议截图保存，放进你的学习笔记。
# 
# 运行下一格后，重点找三块「色块」：**科技双子星（AAPL–MSFT）**、**能源 vs 科技（XLE 那一行/列）**、**各股 vs SPY（和大盘有多像）**。

# %% Notebook cell 12 (code; source_chars=1696)
# ========== 相关性热力图（matplotlib 纯实现）==========
labels = list(corr.columns)                            # 轴标签，顺序与 corr 行列一致
n = len(labels)                                        # 标的数量

fig, ax = plt.subplots(figsize=(8.5, 6.5))             # 创建画布与子图
fig.patch.set_facecolor('#FAFAFA')                   # 画布浅灰背景
ax.set_facecolor('#FFFFFF')                          # 绘图区白色背景

im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)  # 矩阵 → 颜色（红正蓝负）

ax.set_xticks(range(n))                                # 横轴刻度：0 … n-1
ax.set_yticks(range(n))                                # 纵轴刻度
ax.set_xticklabels(labels)                             # 横轴标签 = 股票代码
ax.set_yticklabels(labels)                             # 纵轴标签
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')  # x 标签旋转 45° 防重叠

for i in range(n):                                     # 遍历矩阵行
    for j in range(n):                                 # 遍历矩阵列
        rho = corr.iloc[i, j]                          # 第 (i,j) 格的相关系数
        text_color = 'white' if abs(rho) > 0.55 else 'black'  # 深色背景用白字
        ax.text(                                       # 在格子中心写数值
            j, i, f'{rho:.2f}',                        # 坐标 (列, 行) + 两位小数文本
            ha='center', va='center',                    # 水平/垂直居中对齐
            color=text_color, fontsize=10, fontweight='bold',  # 字体颜色、大小、加粗
        )                                              # text 调用结束

ax.set_title('多标的日收益率相关性热力图（AkShare · 约 3 年）', fontsize=14, pad=12)  # 图标题
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)    # 右侧颜色条（-1 到 1）
plt.tight_layout()                                     # 自动调整边距
plt.show()                                             # 在 Notebook 中显示


# %% [markdown] Notebook cell 13 (markdown)
# #### 📊 读图：热力图怎么读？
# 
# **颜色**
# 
# | 颜色 | 含义 |
# |------|------|
# | 深红 | ρ 接近 +1，同涨同跌 |
# | 浅/白 | ρ 接近 0，各走各的 |
# | 深蓝 | ρ 接近 −1，反向（股票里少见） |
# 
# **格子里的数字**：每个色块中心的 **两位小数** 就是 ρ，和上一节矩阵一致——图是表的「可视化皮肤」。
# 
# **建议找三个问题：**
# 
# 1. **最红的一格**（除对角线）：哪一对最「双胞胎」？若组合里塞满这种 pair，分散化会打折扣。
# 2. **最蓝/最白的一格**：谁和谁最不同步？它们是分散化的候选。
# 3. **XLE 那一行**：和 AAPL、MSFT 比，颜色是否更「冷」？若是，就亲眼看到了 **行业分散** 的效果。
# 
# > 📸 这张图值得截图——它是你 **Lv.5 组合分析师** 的「关系地图」。

# %% [markdown] Notebook cell 14 (markdown)
# ---
# ### 8.4 等权组合 vs 单票波动
# 
# **等权组合（Equal Weight）** = 每只股票占 **相同权重**。这里用 4 只成分各 **25%**（AAPL + MSFT + JPM + XLE），**不含 SPY**——SPY 留给下一节当大盘参照。
# 
# 组合日收益怎么算？极简版：**把 4 列日收益逐行取平均**——就像 4 个人各出 25% 本金，每天的总盈亏是四人当天的平均表现。
# 
# 若成分 **不完全相关**，组合波动往往 **低于** 最「野」的那只单票——这就是 **分散化红利（Diversification Benefit）** 的数学来源。注意：分散主要 **压波动**，不保证 **提高收益**；有时组合涨得比单票慢，但睡得着觉。
# 
# 下一格会打印年化波动率，并画 **横向柱状图**——绿色柱子是组合，蓝色是单票。找一找：**组合是不是比至少一只单票更矮（更稳）？**

# %% Notebook cell 15 (code; source_chars=1464)
# ========== 等权组合波动 vs 单票 ==========
PORT_TICKERS = ['AAPL', 'MSFT', 'JPM', 'XLE']          # 组合成分（不含 SPY 基准）
port_returns = equal_weight_returns(returns, PORT_TICKERS)  # 等权组合日收益序列

comparison = {}                                        # 空字典，收集年化波动
for t in PORT_TICKERS:                                 # 遍历每只成分股
    comparison[t] = annualize_volatility(returns[t])   # 单票年化波动 σ·√252
comparison['等权组合'] = annualize_volatility(port_returns)  # 组合年化波动

comp = pd.Series(comparison).sort_values()           # 转为 Series 并按波动升序

print('年化波动率对比：')                               # 输出标题
print(comp.map('{:.2%}'.format))                     # 格式化为百分比字符串

fig, ax = plt.subplots(figsize=(8, 4.8))             # 横向柱状图画布
fig.patch.set_facecolor('#FAFAFA')                   # 画布背景
ax.set_facecolor('#FFFFFF')                          # 绘图区背景

colors = [                                             # 为每根柱子分配颜色
    '#34C759' if idx == '等权组合' else '#007AFF'    # 组合绿色，单票蓝色
    for idx in comp.index                              # 按 comp 索引顺序生成颜色列表
]                                                      # 列表推导结束
comp.plot(kind='barh', ax=ax, color=colors, alpha=0.88)  # 水平柱状图

ax.set_xlabel('年化波动率')                            # 横轴标签
ax.set_title('单票 vs 等权组合：波动对比', fontsize=14)  # 图标题
ax.grid(True, axis='x', linestyle='--', alpha=0.35)  # 竖向虚线网格
plt.tight_layout()                                     # 调整边距
plt.show()                                             # 显示图表


# %% [markdown] Notebook cell 16 (markdown)
# #### 📊 读图：波动柱状图说明什么？
# 
# - **横轴**：年化波动率（%），越长代表 **一年里「心跳」越猛**。
# - **蓝色柱**：单票——通常 JPM、XLE 或某只科技股的柱子高低因样本而异。
# - **绿色柱「等权组合」**：四票平均后的波动——关键看它在 **什么位置**。
# 
# **典型现象（你的结果可能略有出入）：**
# 
# - 组合柱往往 **低于** 波动最高的那只单票——这就是 **分散化压波动**。
# - 组合柱 **不一定** 低于每一只单票——若四只里有一只特别「野」，组合会被它拉高，但仍可能低于那只最野的。
# - 组合柱 **不一定** 低于四票波动的 **算术平均**——这才是分散化的数学红利。
# 
# **别只盯波动**：波动低 ≠ 赚得多。下一节看 **净值**——同样要抖，你更想要高波动高回报，还是低波动睡得着？

# %% [markdown] Notebook cell 17 (markdown)
# ---
# ### 8.5 组合净值曲线：和 SPY 同台竞技
# 
# 波动率回答「有多抖」，**净值曲线** 回答「最终走到哪」。
# 
# 我们把 **等权组合** 和 **SPY 大盘** 的买入持有净值画在一起——这是 **第二期最后一幅主图**：
# 
# - **蓝实线**：四票等权组合（AAPL+MSFT+JPM+XLE）
# - **橙虚线**：SPY 标普500
# 
# 谁在上面，说明这段样本里 **累计涨得多**；谁更「锯齿」，说明 **日常波动大**。专业团队常问：**我的组合，是在帮客户赚钱，还是只是在重复大盘？**
# 
# > ⚠️ **教学级简化**：这里没有 **再平衡**（权重会随涨跌漂移）、没有 **手续费/滑点**、没有 **分红税**。真实交易会更复杂——但足够建立 **多标的思维** 和 **基准对比** 的习惯。

# %% Notebook cell 18 (code; source_chars=1459)
# ========== 等权组合 vs SPY 净值（第二期收官图）==========
equity_port = buy_hold_equity(port_returns)            # 等权组合买入持有净值
equity_spy  = buy_hold_equity(returns['SPY'])          # SPY 大盘买入持有净值

fig, ax = plt.subplots(figsize=(12, 5))              # 宽幅画布，便于看长期走势
fig.patch.set_facecolor('#FAFAFA')                   # 画布背景
ax.set_facecolor('#FFFFFF')                          # 绘图区背景

ax.plot(                                               # 绘制等权组合净值曲线
    equity_port.index, equity_port,                    # x=日期，y=净值
    color='#007AFF', linewidth=1.6,                    # 蓝色实线
    label='等权组合 (AAPL+MSFT+JPM+XLE)',             # 图例文字
)                                                      # 第一条 plot 结束
ax.plot(                                               # 绘制 SPY 大盘净值曲线
    equity_spy.index, equity_spy,                      # x=日期，y=SPY 净值
    color='#FF9500', linewidth=1.5, linestyle='--',  # 橙色虚线
    label='SPY 大盘',                                  # 图例文字
)                                                      # 第二条 plot 结束

ax.set_title('等权组合 vs 标普500：净值对比（AkShare · 约 3 年）', fontsize=14)  # 标题
ax.set_ylabel('净值（起点=1）')                        # 纵轴
ax.set_xlabel('日期')                                  # 横轴
ax.legend(loc='upper left', framealpha=0.92)         # 图例：左上角
ax.grid(True, linestyle='--', alpha=0.35)            # 虚线网格
plt.tight_layout()                                     # 调整边距
plt.show()                                             # 显示图表


# %% [markdown] Notebook cell 19 (markdown)
# #### 📊 读图：净值曲线怎么读？
# 
# - **纵轴「净值（起点=1）」**：第一天设为 1，之后随涨跌累积——1.5 表示 **+50%**，0.8 表示 **−20%**。
# - **蓝实线（等权组合）**：四行业混搭的买入持有路径——波动通常比单看 AAPL 更「平滑」。
# - **橙虚线（SPY）**：大盘基准——组合 **长期在它上面** = 这段样本跑赢标普；在下面 = 跑输。
# 
# **读图三问：**
# 
# 1. **终点谁更高？** 累计收益胜负（教学样本，不代表未来）。
# 2. **谁更锯齿？** 日常波动体感——组合常比 SPY 或单票更「顺」。
# 3. **最大回撤区间**：肉眼找曲线 **最深的那一段坑**——联系第七章，想想你是否扛得住。
# 
# > 🎓 **第二期收官**：从「一只 AAPL 有多抖」到「四只配一起能否更稳、能否 beat SPY」——你已经站在 **组合思维** 的门口了。

# %% [markdown] Notebook cell 20 (markdown)
# ---
# ## 🎯 挑战任务（第八章 · 第二期通关）
# 
# 完成下面任务，即可解锁 **Lv.5 组合分析师** 徽章。建议 **新开一个代码格**，按顺序做——写不出没关系，先改参数、看图形变化，本身就是学习。
# 
# ### 🟢 入门（改参数就能做）
# 
# 1. **换标的**：把 `TICKERS` 改成 3 只你熟悉的股票 + SPY（如 `TSLA, NVDA, KO, SPY`），重跑 8.1～8.3，观察热力图颜色有何不同。
# 2. **换样本长度**：把 `PERIOD_YEARS = 3` 改成 `1` 或 `5`，对比相关性是否变化——**短期危机**（如 2020）常让相关性突然升高（「万物同跌」）。
# 3. **截图打卡**：保存 **热力图 + 波动柱状图 + 净值对比图** 三张图，写一句「我发现了 XXX 和 YYY 相关最高」。
# 
# ### 🟡 进阶（需要写几行代码）
# 
# 4. **找低相关一对**：在 `corr` 矩阵里找出 **ρ 最低** 的两只（代码已自动打印），思考：若只买这两只能否「一涨一跌」？实际上长期 ρ 很少接近 −1，但 **相对更低** 就有分散价值。
# 5. **不等权实验**：构造 `0.5*AAPL + 0.5*MSFT` 的日收益，和四票等权比 **年化波动**——哪个更高？为什么「押注两个科技龙头」反而可能更抖？
# 6. **三票组合**：只选 AAPL、JPM、XLE 做等权，对比四票等权——**多加一只高度相关的 MSFT**，组合波动是升还是降？
# 7. **算组合 vs 单票平均**：打印 `PORT_TICKERS` 四票波动率的 **简单平均**，和等权组合波动对比——组合波动是否 **低于** 平均？（提示：这就是分散化的数值证据。）
# 
# ### 🔴 挑战（烧脑但超有收获）
# 
# 8. **滚动相关**：用 `returns['AAPL'].rolling(60).corr(returns['MSFT'])` 画 60 日滚动相关——**危机时相关会飙升吗？** 这对「分散失效」有什么启示？
# 9. **手写组合波动（近似）**：查公式 σ²_p ≈ Σ w_i² σ_i² + 2Σ w_i w_j σ_i σ_j ρ_ij，用 2 只标的（各 50%）手算组合年化波动，和 `annualize_volatility` 结果对照（允许小误差）。
# 10. **基准超额**：计算等权组合与 SPY 的 **累计收益差**（末净值相减），写 3 句话：这段样本里组合是跑赢还是跑输大盘？波动对比如何？你会更愿意持有哪条曲线？
# 11. **思考题**：若所有股票 ρ 都变成 1（完美同涨同跌），等权组合的波动会等于哪只单票？分散化还有意义吗？
# 12. **复盘第二期**：用 bullet 列出第二～八章你最喜欢的 **一个指标 + 一张图**，说明它如何帮你做投资决策（不必很长，但要具体）。
# 
# > 🏁 **通关标准**：至少完成 **入门 3 项 + 进阶 2 项**；若完成任意 **挑战项**，可在笔记里给自己标注 **Lv.5+**。

# %% Notebook cell 21 (code; source_chars=0)


# %% Notebook cell 22 (code; source_chars=0)


# %% Notebook cell 23 (code; source_chars=0)


# %% [markdown] Notebook cell 24 (markdown)
# ## 本章总结
# 
# - **多标的分析** 从「对齐数据表」开始：列是标的，行是日期，`.dropna()` 保证公平对比。
# - **相关性 ρ** 量化「同涨同跌」；热力图让关系 **一眼可读**——科技内部往往很红，能源常与科技不同步。
# - **等权组合** 在标的不完全相关时，波动往往 **低于** 单票——分散投资的数学直觉，不是理财鸡汤。
# - **SPY 基准** 帮你回答：我的组合是在 **创造差异**，还是 **重复大盘**？
# 
# ---
# 
# **第二期知识地图（你已走完）：**
# 
# | 章节 | 核心问题 |
# |------|---------|
# | 第5章 波动率 | 有多抖？ |
# | 第6章 Sharpe / Beta | 性价比如何？相对大盘多敏感？ |
# | 第7章 回撤 / 仓位 | 最惨能亏多少？买多少合适？ |
# | 第8章 组合 / 相关 | 多买几只，真的更稳吗？ |
# 
# **第三期预告（规划）**：因子选股、权重优化（不止等权）、以及 AI 量化入门——从「看关系」走向「系统性地配权重」。Stay tuned.

