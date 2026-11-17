# -*- coding: utf-8 -*-
# 来源：Quant-for-babies / ThreshoLdnnis
# 原始 Notebook：notebooks/phase2_intro/02_夏普比率与Beta.ipynb
# 来源链接：https://github.com/ThreshoLdnnis/Quant-for-babies/blob/711ca83a82f6f1bb1745c6ca28c1ed3ecd2a97bd/notebooks/phase2_intro/02_夏普比率与Beta.ipynb
# Git blob SHA：a2d5d30bbada883712a7a853ef4939abea171b5d
# 导出说明：代码单元及原有注释原样保留；Markdown 单元转换为注释。
# 使用方式：按 # %% 分隔的单元顺序运行；环境及第 3 章数据源选择见 README.md。

# %% [markdown] Notebook cell 1 (markdown)
# # 《零基础学习量化金融》
# ## 从Python到AI量化交易实战 第二期
# ### 第6章：夏普比率与 Beta

# %% [markdown] Notebook cell 2 (markdown)
# ---
# 
# ## 本章你将学会
# 
# - ✅ 为什么 **只看收益率不够**（高风险高收益的陷阱）
# - ✅ 计算 **夏普比率（Sharpe Ratio）** 并解读
# - ✅ 理解 **Beta**：股票相对大盘的「敏感度」
# - ✅ 用 **numpy** 算协方差，手推 Beta
# - ✅ 给多只股票做 **风险收益散点图**
# 
# **当前等级**
# 🎮 **Lv.4 风险分析师**
# 
# **本章难度**
# ⭐⭐⭐☆☆
# 
# **预计学习时间**
# 30～45 分钟（需联网）
# 
# **前置知识**
# 
# - 完成 **第五章**（波动率、年化）
# - 第一期 **第四章**（回测与基准对比）
# 
# ---
# 
# 假设两只股票：A 一年赚 30%，B 一年赚 15%。你会选 A 吗？
# 
# 如果 A 的波动是 B 的 **三倍**，很多专业投资者反而更选 B——因为 A 的「性价比」可能更差。
# 
# 这一章，我们学习两个最常用的 **风险指标**：**夏普比率** 和 **Beta**。
# 

# %% [markdown] Notebook cell 3 (markdown)
# ### 环境准备
# 
# 如果提示缺少库，在项目根目录执行：`pip install -r requirements.txt`
# 
# > 本章行情数据改用 **AkShare**（`stock_us_daily`）拉取美股，无需 yfinance。

# %% Notebook cell 4 (code; source_chars=2090)
# ========== 环境准备：导入库 + 量化小工具 ==========
import warnings                                   # 导入警告控制模块
warnings.filterwarnings('ignore')              # 隐藏次要警告，Notebook 输出更干净

import statistics as stats                      # 标准库：均值、中位数、标准差
import numpy as np                              # 数值计算（协方差、开方等）
import pandas as pd                             # 表格数据处理
import matplotlib.pyplot as plt                 # 绘图
import akshare as ak                            # 国内金融数据接口（本章拉美股）
import time                                     # 批量请求间隔，避免触发接口限速

plt.rcParams['font.sans-serif'] = [              # 跨平台中文字体回退
    'PingFang SC', 'Microsoft YaHei', 'SimHei',
    'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans',
]
plt.rcParams['axes.unicode_minus'] = False      # 坐标轴负号正常显示

TRADING_DAYS = 252                              # 美股常用年化交易日数


def annualize_return(daily_returns: pd.Series) -> float:
    """日收益率 → 年化收益率（复利近似）。"""
    mean_daily = daily_returns.mean()           # 日收益算术平均
    return (1 + mean_daily) ** TRADING_DAYS - 1  # 复利外推至 252 个交易日


def annualize_volatility(daily_returns: pd.Series) -> float:
    """日收益率 → 年化波动率。"""
    daily_std = daily_returns.std()             # 日收益标准差
    return daily_std * np.sqrt(TRADING_DAYS)    # 日波动 × √252 → 年化波动


def sharpe_ratio(daily_returns: pd.Series, rf_annual: float) -> float:
    """简化夏普：(年化收益 − 无风险利率) / 年化波动。"""
    ann_ret = annualize_return(daily_returns)   # 先算年化收益
    ann_vol = annualize_volatility(daily_returns)  # 再算年化波动
    excess = ann_ret - rf_annual                # 超额收益 = 收益 − 无风险
    return excess / ann_vol                     # 每单位风险对应的超额回报


def beta_vs_market(stock: pd.Series, market: pd.Series) -> tuple[float, float]:
    """numpy 与 pandas 双算法计算 Beta，便于交叉验证。"""
    cov = np.cov(stock, market)                 # 2×2 协方差矩阵 [[Var股, Cov], [Cov, Var市]]
    beta_numpy = cov[0, 1] / cov[1, 1]          # Cov(股, 市) / Var(市)
    beta_pandas = stock.cov(market) / market.var()  # pandas 等价写法
    return beta_numpy, beta_pandas              # 返回双结果供对照


print('环境就绪 ✓')                               # 提示：环境加载完成


# %% [markdown] Notebook cell 5 (markdown)
# ---
# 
# ### 6.1 夏普比率：性价比
# ## 基础定义与背景
# 夏普比率，也称作夏普指数，是标准化的投资组合绩效评价指标，与特雷诺比率、詹森指数并列为基金绩效评价三大经典衡量指标。
# 该指标由诺贝尔经济学奖得主威廉·夏普于1966年首次提出，目前是全球范围内衡量资产、基金组合风险调整收益最通用的标准化评价工具。
# 
# ## 核心公式与经济含义
# ### 计算公式
# $$
# \text{Sharpe Ratio} = \frac{E(R_p)-R_f}{\sigma_p}
# $$
# - $E(R_p)$：投资组合预期收益率
# - $R_f$：无风险利率
# - $\sigma_p$：投资组合收益率标准差（代表组合总波动风险）
# 
# ### 指标内涵
# 夏普比率同时综合考量投资的收益与风险，数值代表**每承担一单位总波动风险，能够获取的超额无风险收益**。
# 
# ## 适用场景与固有局限
# ### 适用范围
# 用于横向对比不同投资组合经风险调整后的真实收益水平，衡量单位风险的超额回报。
# 
# ### 应用缺陷
# 1. 指标将收益率全部波动视作风险，无法区分上行盈利波动与下行亏损波动；
# 2. 理论假设资产收益率服从正态分布，与金融市场真实厚尾行情存在偏差；
# 3. 常规测算依赖历史行情数据，对未来收益的预测存在滞后性。
# 
# ## 指标拓展与现实应用
# 自1966年提出后，学界衍生出多种改良版本，包含修正夏普指数、基于风险价值VaR构建的VaR-Sharpe绩效模型等，弥补传统夏普比率的天然缺陷。
# 
# **市场实例**
# 2024年3月，高盛发布观点维持对A股、港股的高配配置建议，判断短期A股资产具备更高夏普比率，风险调整收益优于其他大类资产。
# 
# 
# 
# **夏普比率** 回答一个问题：
# 
# > 每多承担一单位风险，我比「无风险收益」多赚了多少？
# 
# 简化版公式（for beginners）：
# 
# $$
# \text{Sharpe} \approx \frac{\text{年化收益} - \text{无风险利率}}{\text{年化波动率}}
# $$
# 
# 👉 **[🎬 点击这里：观看夏普比率动态演示](../../assets/interactive/sharpe-ratio-demo.html)**（新页面打开；逐步代入年化收益、无风险利率 4%、年化波动，算出 Sharpe）
# 
# 无风险利率可先用 **常数近似**（例如 4%），后面章节再细化。
# 
# **直觉**：Sharpe 越高，同样风险下赚得越多——像「性价比」。
# 

# %% Notebook cell 6 (code; source_chars=3148)
# ========== 下载行情 → 日收益 → 年化指标与 Sharpe ==========
# 数据链：akshare → ak.stock_us_daily() → 新浪财经美股日线（联网，非模拟）

tickers = [                                      # 共 10 只：9 只个股 + 1 只大盘 ETF
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',    # 科技巨头
    'NVDA', 'TSLA', 'AMD',  'JPM',              # 高波动成长 + 银行对照
    'SPY',                                       # 标普 500 ETF，作 Beta 基准
]
period_years = 3                                 # 回溯约 3 年
rf_annual    = 0.04                              # 无风险利率近似 4%

start_date = (                                   # 计算样本起始日
    pd.Timestamp.today()                         # 今天
    - pd.DateOffset(years=period_years, days=30) # 往前 3 年，多留 30 天缓冲
).strftime('%Y-%m-%d')                           # 格式化为 'YYYY-MM-DD'

print(f'akshare 版本: {ak.__version__}')         # 打印库版本，便于排查
print('接口: ak.stock_us_daily(symbol, adjust="qfq")  # 新浪财经 · 前复权')


def fetch_us_close(symbol: str, start: str) -> pd.Series:
    """联网拉取单只美股前复权收盘价，并按起始日裁剪。"""
    df = ak.stock_us_daily(symbol=symbol, adjust='qfq')  # 发起 HTTP 请求
    if df is None or df.empty:                     # 网络/接口异常时主动报错
        raise RuntimeError(f'{symbol} 未返回数据，请检查网络或 akshare 版本')

    df['date'] = pd.to_datetime(df['date'])        # 字符串 → 日期类型
    close = (                                      # 整理为以日期为索引的收盘价序列
        df.set_index('date')                       # 日期列设为行索引
          .sort_index()                             # 按时间升序排列
          ['close']                                 # 只保留收盘价列
          .rename(symbol)                           # Series 命名 = 股票代码
    )
    return close.loc[close.index >= start]         # 截取 start 之后的样本


print(f'\n正在下载 {len(tickers)} 只美股（约 {period_years} 年）…')
frames = {}
for t in tickers:
    frames[t] = fetch_us_close(t, start_date)
    time.sleep(1)                                # 新浪接口限速，连续请求间隔 1 秒
prices = pd.DataFrame(frames).dropna()           # 宽表：每列一只股票

returns = prices.pct_change().dropna()             # 简单日收益率 r_t = P_t/P_{t-1} − 1

api_tail = ak.stock_us_daily(symbol='AAPL', adjust='qfq').tail(1)  # 再拉一次 API 末行作核验
print('\n【核验】AAPL 接口原始末行（AkShare 直接返回）：')
print(api_tail)                                  # Notebook 内美观显示表格
print(f'样本区间：{prices.index[0].date()} → {prices.index[-1].date()}，共 {len(returns)} 个交易日')
print(f'AAPL 末收（对齐后）: {prices["AAPL"].iloc[-1]:.2f}')  # 应与 api_tail 的 close 一致

metrics = pd.DataFrame({                           # 数值版指标表（供后续作图）
    t: {                                           # 每列对应一只 ticker
        '年化收益': annualize_return(returns[t]),
        '年化波动': annualize_volatility(returns[t]),
        'Sharpe': sharpe_ratio(returns[t], rf_annual),
    }
    for t in tickers                               # 遍历全部 10 只
}).T.rename_axis('股票')                            # 转置后行 = 股票代码

summary = metrics.copy()                           # 复制一份用于格式化展示
summary['年化收益'] = summary['年化收益'].map('{:.2%}'.format)  # 收益 → 百分比字符串
summary['年化波动'] = summary['年化波动'].map('{:.2%}'.format)  # 波动 → 百分比字符串
summary['Sharpe']   = summary['Sharpe'].map('{:.2f}'.format)    # Sharpe 保留两位小数

print('\n风险收益一览（近似 Sharpe）：')
print(summary)                                   # 展示格式化后的汇总表


# %% [markdown] Notebook cell 7 (markdown)
# ---
# 
# ### 6.2 风险收益散点图
# 
# 把每只股票画在 **横轴=波动、纵轴=收益** 的平面上，一眼看出：
# 
# - 左上角：低波动、高收益（理想，但少见）
# - 右下角：高波动、低收益（性价比差）
# 
# **SPY** 代表大盘，常作对比基准。
# 

# %% Notebook cell 8 (code; source_chars=2316)
# ========== 风险-收益散点图（横轴波动 · 纵轴收益 · 10 只标的）==========
from matplotlib.ticker import FuncFormatter       # 坐标轴百分比格式化工具

PALETTE = {                                         # 10 只股票各配一色，图例清晰
    'AAPL':  '#007AFF', 'MSFT':  '#00A4EF', 'GOOGL': '#4285F4',
    'AMZN':  '#FF9900', 'META':  '#0668E1', 'NVDA':  '#76B900',
    'TSLA':  '#E82127', 'AMD':   '#ED1C24', 'JPM':   '#006747',
    'SPY':   '#FF9500',
}

fig, ax = plt.subplots(figsize=(10, 7))            # 10 个点，画布略放大
fig.patch.set_facecolor('#FAFAFA')                # 画布浅灰底
ax.set_facecolor('#FFFFFF')                       # 绘图区白底

for ticker, row in metrics.iterrows():            # 复用上格 metrics，避免重复计算
    vol = row['年化波动']                          # 横坐标：年化波动
    ret = row['年化收益']                          # 纵坐标：年化收益
    color = PALETTE.get(ticker, '#666666')        # 取调色板颜色，缺省灰色
    ax.scatter(
        vol, ret,                                 # 单点坐标
        s=140,                                    # 点的大小
        c=color,                                  # 填充色
        edgecolors='white',                       # 白边，点与点更易区分
        linewidths=1.0,
        label=ticker,                             # 图例标签
        zorder=3,                                 # 图层顺序（点在网格之上）
        alpha=0.90,                               # 轻微透明
    )
    ax.annotate(
        ticker,                                   # 标注文字 = 代码
        (vol, ret),                               # 锚点坐标
        textcoords='offset points',               # 偏移量单位 = 像素点
        xytext=(6, 5),                            # 文字相对锚点的偏移
        fontsize=9,                               # 10 只时字号略小，防重叠
        fontweight='bold',
        color=color,
    )

pct = FuncFormatter(lambda x, _pos: f'{x:.0%}')     # 0.25 显示为 25%
ax.xaxis.set_major_formatter(pct)                 # 横轴百分比格式
ax.yaxis.set_major_formatter(pct)                 # 纵轴百分比格式

ax.set_xlabel('年化波动率', fontsize=12)           # 横轴标题
ax.set_ylabel('年化收益率', fontsize=12)           # 纵轴标题
ax.set_title('风险-收益散点图（10 只 · 约 3 年 · AkShare 行情）', fontsize=14, pad=12)
ax.grid(True, linestyle='--', alpha=0.35)         # 虚线网格，轻量不抢戏
ax.legend(loc='upper left', ncol=2, fontsize=9, framealpha=0.9)  # 两列图例省空间
plt.tight_layout()                                # 自动调整边距
plt.show()                                        # 在 Notebook 中显示图表


# %% [markdown] Notebook cell 9 (markdown)
# ---
# 
# ### 6.3 Beta：相对大盘的敏感度
# 
# **Beta（β）** 衡量：大盘动 1%，这只股票平均动多少？
# 
# - β ≈ 1：跟大盘差不多
# - β > 1：比大盘更「冲」（涨跌幅放大）
# - β < 1：比大盘更「稳」
# 
# 计算公式（用日收益率）：
# 
# $$
# \beta = \frac{\text{Cov}(R_{\text{股票}}, R_{\text{大盘}})}{\text{Var}(R_{\text{大盘}})}
# $$
# 
# 👉 **[🎬 点击这里：观看 Beta 动态演示](../../assets/interactive/beta-demo.html)**（新页面打开；逐步代入 Cov、Var，算出 β，可切换 9 只个股对比 SPY）
# 
# 
# 下面用 **numpy** 的 `cov` 手算，再与 pandas 对照。
# 

# %% Notebook cell 10 (code; source_chars=785)
# ========== Beta：相对 SPY 的敏感度（numpy ⟷ pandas 对照）==========
MARKET = returns['SPY']                            # 大盘基准（标普 500 ETF）
STOCKS = [t for t in tickers if t != 'SPY']       # 其余 9 只个股，逐一算 Beta

beta_rows = []                                     # 收集每只股票的 Beta 结果
for ticker in STOCKS:                              # 遍历 9 只个股
    stock = returns[ticker]                        # 该股日收益率序列
    b_np, b_pd = beta_vs_market(stock, MARKET)     # numpy / pandas 双算法
    beta_rows.append({                             # 追加一行结果
        '股票': ticker,
        'Beta(numpy)':  b_np,
        'Beta(pandas)': b_pd,
    })

beta_df = pd.DataFrame(beta_rows).set_index('股票')  # 转为表格，股票代码作行索引

print('Beta 相对 SPY（β>1 比大盘更冲，β<1 更稳）：')
print(beta_df.map('{:.2f}'.format))              # 保留两位小数展示


# %% [markdown] Notebook cell 11 (markdown)
# ---
# 
# ### 6.4 用 statistics 看收益分布
# 
# **statistics** 是 Python 标准库，不需要额外安装。除了均值、标准差，还可以看 **中位数**——当某天极端涨跌把均值「拉歪」时，中位数更稳健。
# 
# 金融数据分析 ≠ 只会 pandas；标准库在快速验证时很方便。
# 

# %% Notebook cell 12 (code; source_chars=517)
# ========== statistics 标准库：AAPL 日收益分布快照 ==========
aapl_daily = returns['AAPL'].dropna().tolist()     # Series → list，供 statistics 使用

print('── AAPL 日收益率分布（statistics 标准库）──')  # 分隔标题
print(f'均值:       {stats.mean(aapl_daily):>8.4%}')    # 算术平均（易受极端值影响）
print(f'中位数:     {stats.median(aapl_daily):>8.4%}')  # 50% 分位，更抗极端值
print(f'标准差:     {stats.pstdev(aapl_daily):>8.4%}')  # 总体标准差（除以 N）
print(f'最大单日涨: {max(aapl_daily):>8.2%}')           # 样本内最佳单日表现
print(f'最大单日跌: {min(aapl_daily):>8.2%}')           # 样本内最差单日表现


# %% [markdown] Notebook cell 13 (markdown)
# ---
# 
# ## 🎯 挑战任务（第六章通关）
# 
# 1. **Sharpe 排序**：对上面 10 只标的，按 Sharpe 从高到低排序并打印前三名。
# 2. **Beta 实验**：比较 `JPM`（银行）与 `NVDA` 的 Beta，谁更贴近大盘 SPY？
# 3. **思考题**：Sharpe 高是否意味着未来一定更好？（提示：回顾第一期「回测不是算命」）
# 

# %% Notebook cell 14 (code; source_chars=0)


# %% [markdown] Notebook cell 15 (markdown)
# ## 本章总结
# 
# - **夏普比率** = 风险调整后的「性价比」；不能只看绝对收益。
# - **Beta** 描述相对大盘的敏感度；用协方差 / 大盘方差即可手算。
# - **散点图** 是沟通风险收益最直观的方式之一。
# - `numpy.cov`、`statistics`、`pandas` 可以交叉验证同一指标。
# 
# **下一章预告**：我们会深入 **最大回撤**，并初探 **仓位管理**——同样策略，买多少可能决定生死。
# 

