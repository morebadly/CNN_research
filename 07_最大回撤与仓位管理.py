# -*- coding: utf-8 -*-
# 来源：Quant-for-babies / ThreshoLdnnis
# 原始 Notebook：notebooks/phase2_intro/03_最大回撤与仓位管理.ipynb
# 来源链接：https://github.com/ThreshoLdnnis/Quant-for-babies/blob/711ca83a82f6f1bb1745c6ca28c1ed3ecd2a97bd/notebooks/phase2_intro/03_最大回撤与仓位管理.ipynb
# Git blob SHA：9fe41c944fafabf7d4c39e10b64e38fa965c9a7d
# 导出说明：代码单元及原有注释原样保留；Markdown 单元转换为注释。
# 使用方式：按 # %% 分隔的单元顺序运行；环境及第 3 章数据源选择见 README.md。

# %% [markdown] Notebook cell 1 (markdown)
# # 《零基础学习量化金融》
# ## 从Python到AI量化交易实战 第二期
# ### 第7章：最大回撤与仓位管理

# %% [markdown] Notebook cell 2 (markdown)
# ---
# 
# ## 本章你将学会
# 
# - ✅ 从第一期回测出发，**深入理解最大回撤**
# - ✅ 画 **水下曲线（Underwater Plot）** 看「坑有多深、多久爬出来」
# - ✅ 理解 **仓位（Position Sizing）** 为什么和策略同样重要
# - ✅ 用代码对比 **满仓 vs 半仓** 对净值曲线的影响
# - ✅ 建立「先想能亏多少，再想能赚多少」的风险意识
# 
# **当前等级**
# 🎮 **Lv.5 仓位管理员**
# 
# **本章难度**
# ⭐⭐⭐☆☆
# 
# **预计学习时间**
# 30～45 分钟（需联网）
# 
# **前置知识**
# 
# - 完成 **第一期第四章**（回测、最大回撤初识）
# - 完成 **第五～六章**（波动率、Sharpe）
# 
# ---
# 
# 第五、六章我们学会了用 **波动率** 和 **Sharpe** 衡量「风险与性价比」。但数字再漂亮，账户一旦深跌，人往往先扛不住——**最大回撤** 衡量的正是这段最煎熬的经历。
# 
# 第一期你已经见过最大回撤这个数字。这一章我们把它 **拆开、画出来、和仓位挂钩**：你会看到 AAPL 买入持有究竟「坑」有多深，再亲手验证：**同一套双均线规则，只改仓位，曲线就会完全不同。**
# 
# 专业交易员常说：**策略决定方向，仓位决定生死。** 本章不追求复杂模型，先把这条直觉刻进肌肉记忆。
# 

# %% [markdown] Notebook cell 3 (markdown)
# ### 环境准备
# 
# 如果提示缺少库，在项目根目录执行：`pip install -r requirements.txt`
# 
# > 本章行情改用 **AkShare**（`stock_us_daily`）拉取美股，无需 yfinance。
# 
# 本格还预先定义了 **`fetch_us_close`、`compute_drawdown`、`max_drawdown`** 等工具函数——后面每一节都会复用，避免重复写公式。

# %% Notebook cell 4 (code; source_chars=2132)
# ========== 环境准备：导入库 + 回撤 / 仓位工具函数 ==========
import warnings                                   # 导入警告控制模块
warnings.filterwarnings('ignore')              # 隐藏次要警告，输出更干净

import statistics as stats                      # 标准库：均值、中位数等
import numpy as np                              # 数值计算
import pandas as pd                             # 表格数据处理
import matplotlib.pyplot as plt                 # 绘图
import akshare as ak                            # 国内金融数据接口（本章拉美股）

plt.rcParams['font.sans-serif'] = [              # 跨平台中文字体回退
    'PingFang SC', 'Microsoft YaHei', 'SimHei',
    'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans',
]
plt.rcParams['axes.unicode_minus'] = False      # 坐标轴负号正常显示

TRADING_DAYS = 252                              # 美股常用年化交易日数


def fetch_us_close(symbol: str, years: float = 3.0) -> pd.Series:
    """联网拉取美股前复权收盘价（AkShare → 新浪财经）。"""
    start = (
        pd.Timestamp.today()
        - pd.DateOffset(years=years, days=30)   # 多留缓冲，避免边界缺数据
    ).strftime('%Y-%m-%d')
    df = ak.stock_us_daily(symbol=symbol, adjust='qfq')  # 发起 HTTP 请求
    if df is None or df.empty:                   # 网络异常时主动报错
        raise RuntimeError(f'{symbol} 未返回数据，请检查网络或 akshare 版本')
    df['date'] = pd.to_datetime(df['date'])      # 字符串 → 日期
    close = (
        df.set_index('date')                     # 日期作索引
          .sort_index()                          # 时间升序
          ['close']                              # 只保留收盘价
          .loc[lambda s: s.index >= start]       # 截取样本区间
          .rename(symbol)                        # Series 命名 = 代码
    )
    return close


def buy_hold_equity(daily_returns: pd.Series) -> pd.Series:
    """买入持有：日收益连乘 → 净值曲线（起点 ≈ 1）。"""
    return (1 + daily_returns).cumprod()


def compute_drawdown(equity: pd.Series) -> pd.Series:
    """回撤序列 = 当前净值 / 历史最高净值 − 1（≤ 0）。"""
    running_peak = equity.cummax()               # 截至当日的历史最高净值
    return equity / running_peak - 1             # 相对高点的跌幅


def max_drawdown(equity: pd.Series) -> float:
    """最大回撤 = 回撤序列中的最小值（最深的那次坑）。"""
    return compute_drawdown(equity).min()


print('环境就绪 ✓')                               # 提示：环境加载完成

# %% Notebook cell 5 (code; source_chars=0)


# %% [markdown] Notebook cell 6 (markdown)
# ---
# 
# ### 7.1 最大回撤：账户最深的坑
# 
# **回撤（Drawdown）** = 从历史最高点往下掉了多少比例。可以把它想成：你站在山顶（净值前高），现在往下走了多远。
# 
# **最大回撤（Max Drawdown）** = 整段样本里，最深的那一次「坑」。它通常是一个 **负数**，绝对值越大，说明曾经跌得越惨。
# 
# 例子：净值从 1.5 跌到 1.0，回撤 = (1.0 − 1.5) / 1.5 = **−33%**。
# 
# 和第六章 Sharpe 不同，最大回撤回答的问题不是「赚多少」，而是：**最惨时你会经历什么？** 如果你无法在 −30% 的浮亏里坚持持仓，再高的长期收益也可能与你无关——因为你会在中间某一刻割肉离场。
# 
# 下面用 AAPL 近三年的 **买入持有** 净值，算出真实的最大回撤数值。
# 

# %% Notebook cell 7 (code; source_chars=886)
# ========== 下载 AAPL → 买入持有净值 → 最大回撤 ==========
TICKER       = 'AAPL'                            # 演示标的
PERIOD_YEARS = 3                                 # 回溯约 3 年

print(f'akshare 版本: {ak.__version__}')
print('接口: ak.stock_us_daily(symbol, adjust="qfq")  # 新浪财经 · 前复权')

prices  = fetch_us_close(TICKER, PERIOD_YEARS)   # 收盘价序列（DatetimeIndex）
returns = prices.pct_change().dropna()           # 简单日收益率 r_t = P_t/P_{t-1} − 1
equity  = buy_hold_equity(returns)               # 买入持有净值曲线

drawdown = compute_drawdown(equity)              # 逐日回撤序列（≤ 0）
max_dd   = max_drawdown(equity)                  # 整段样本最深回撤
worst_dt = drawdown.idxmin()                     # 回撤最深的那一天

print(f'\n样本区间：{prices.index[0].date()} → {prices.index[-1].date()}，共 {len(returns)} 个交易日')
print(f'{TICKER} 买入持有 · 最大回撤: {max_dd:.2%}')
print(f'回撤最深日期: {worst_dt.date()}')
print(f'末收（AkShare）: {prices.iloc[-1]:.2f}')


# %% [markdown] Notebook cell 8 (markdown)
# ---
# 
# ### 7.2 水下曲线（Underwater Plot）
# 
# 光有一个「最大回撤 = −33%」的数字还不够直观。把 **逐日回撤序列** 画成填充图，就是 **水下曲线**——净值在水面（前高）之下时，填充区域越深，说明离历史高点越远。
# 
# 读图时可以关注两件事：
# 
# 1. **深度**：坑有多深（和最大回撤对应）
# 2. **宽度**：在水下待了多久才爬回水面——同样 −20% 的回撤，三个月恢复和三年恢复，心理压力完全不同
# 
# 好的策略不仅要回撤浅，还要 **恢复快**。后面讲因子、组合时，我们会继续用这条曲线做对比。
# 

# %% Notebook cell 9 (code; source_chars=1301)
# ========== 净值 + 水下曲线（Underwater Plot）==========
running_peak = equity.cummax()                    # 历史最高净值（用于上图虚线）

PALETTE = dict(equity='#007AFF', underwater='#E82127')  # 统一配色

fig, (ax1, ax2) = plt.subplots(
    2, 1,                                       # 上下两个子图
    figsize=(12, 7),
    sharex=True,                                # 共用横轴（日期）
    gridspec_kw={'height_ratios': [2, 1]},      # 上图占 2/3 高度
)
fig.patch.set_facecolor('#FAFAFA')             # 画布浅灰底

ax1.set_facecolor('#FFFFFF')
ax1.plot(equity.index, equity, color=PALETTE['equity'], linewidth=1.6, label='买入持有净值')
ax1.plot(running_peak.index, running_peak, color='gray', linestyle='--', alpha=0.55, label='历史最高')
ax1.set_title(f'{TICKER} 买入持有净值 & 回撤（水下曲线 · AkShare）', fontsize=14)
ax1.set_ylabel('净值（起点=1）')
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.35)

ax2.set_facecolor('#FFFFFF')
ax2.fill_between(drawdown.index, drawdown, 0, color=PALETTE['underwater'], alpha=0.45)
ax2.axhline(max_dd, color='#333333', linestyle=':', linewidth=1.2, label=f'最大回撤 {max_dd:.1%}')
ax2.set_ylabel('回撤')
ax2.set_xlabel('日期')
ax2.legend(loc='lower left')
ax2.grid(True, linestyle='--', alpha=0.35)

plt.tight_layout()                              # 自动调整边距
plt.show()                                      # 在 Notebook 中显示


# %% [markdown] Notebook cell 10 (markdown)
# ---
# 
# ### 7.3 仓位管理：同样策略，不同结果
# 
# **仓位（Position Sizing）** = 你投入多少比例的资金去跟随信号。
# 
# 很多初学者只优化「什么时候买、什么时候卖」，却忽略 **买多少**。其实，在简化模型下，仓位是调节风险最直接的旋钮——**同一套 MA 金叉死叉，满仓和半仓走的是两条不同的人生。**
# 
# 本章用的简化规则：
# 
# | 仓位 | 含义 | 直觉 |
# |------|------|------|
# | **满仓** | 100% 资金跟着策略 | 收益、回撤都「满格」 |
# | **半仓** | 50% 资金参与 | 策略日收益 × 0.5，回撤大约也减半 |
# 
# 真实交易还要考虑手续费、滑点、杠杆、加仓规则——这里先建立直觉。下一格我们用 **5/20 双均线** 在 AAPL 上跑一遍，并打印三种净值的最大回撤。
# 

# %% Notebook cell 11 (code; source_chars=1101)
# ========== 双均线策略：满仓 vs 半仓 ==========
MA_SHORT, MA_LONG = 5, 20                        # 短/长均线窗口（交易日）
HALF_SIZE         = 0.5                          # 半仓 = 50% 资金参与

df = pd.DataFrame({'Close': prices, 'Return': returns}).dropna()  # 对齐价格与收益
df['MA_S'] = df['Close'].rolling(MA_SHORT).mean()   # 短期均线
df['MA_L'] = df['Close'].rolling(MA_LONG).mean()    # 长期均线
df['Signal'] = (df['MA_S'] > df['MA_L']).astype(float)  # 1=多头, 0=空仓
df['StratRet'] = df['Signal'].shift(1) * df['Return']     # 次日按昨日信号交易（无未来函数）
df = df.dropna()                                 # 去掉 MA 暖机期的 NaN

df['Equity_100'] = buy_hold_equity(df['StratRet'])              # 满仓净值
df['Equity_50']  = buy_hold_equity(df['StratRet'] * HALF_SIZE) # 半仓：收益按仓位缩放
df['Equity_BH']  = buy_hold_equity(df['Return'])               # 买入持有对照

scenarios = [                                    # 统一遍历，避免重复代码
    ('双均线 · 满仓', 'Equity_100'),
    ('双均线 · 半仓', 'Equity_50'),
    ('买入持有',      'Equity_BH'),
]

print('最大回撤对比：')
for label, col in scenarios:
    dd = max_drawdown(df[col])                   # 复用上文工具函数
    print(f'  {label:12s}: {dd:.2%}')


# %% [markdown] Notebook cell 12 (markdown)
# ---
# 
# ### 7.4 三条净值曲线对比
# 
# 上一格已经算出了数字，这里把 **双均线满仓、双均线半仓、买入持有** 三条净值画在一起。
# 
# 读图时建议对照三个问题：
# 
# - 谁涨得最高？（最终收益）
# - 谁跌得最惨？（最大回撤，回看上一格打印结果）
# - 曲线是否更「平滑」？（半仓往往波动更小）
# 
# 同一段行情、同一套 MA 规则——**只改仓位**，曲线形态就会变。半仓可能 **少赚**，但也 **少亏、少回撤**。没有免费午餐，只有适不适合你的风险预算。
# 

# %% Notebook cell 13 (code; source_chars=776)
# ========== 三条净值曲线：仓位如何改变命运 ==========
STYLE = {                                        # 线型 / 颜色 / 粗细
    'Equity_100': dict(color='#007AFF', lw=1.6, ls='-',  label='双均线 · 满仓'),
    'Equity_50':  dict(color='#34C759', lw=1.6, ls='-',  label='双均线 · 半仓'),
    'Equity_BH':  dict(color='#8E8E93', lw=1.3, ls='--', label='买入持有'),
}

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FFFFFF')

for col, kw in STYLE.items():                    # 同一循环画三条曲线
    ax.plot(df.index, df[col], **kw)

ax.set_title(f'{TICKER} 双均线（{MA_SHORT}/{MA_LONG}）· 仓位对比 · AkShare 行情', fontsize=14)
ax.set_ylabel('净值')
ax.set_xlabel('日期')
ax.legend(loc='upper left', framealpha=0.92)
ax.grid(True, linestyle='--', alpha=0.35)

plt.tight_layout()
plt.show()


# %% [markdown] Notebook cell 14 (markdown)
# ---
# 
# ## 🎯 挑战任务（第七章通关）
# 
# 1. **换策略参数**：把 MA 改成 10/30，比较满仓与 30% 仓位的最大回撤。观察：参数变了，半仓是否仍然明显「减坑」？
# 2. **算恢复时间**：找到最大回撤发生日，看之后 **60 个交易日** 内净值是否回到前高。（提示：可用 `running_peak` 与 `equity` 对比，或先看水下曲线宽度）
# 3. **思考题**：如果你只能承受 −15% 回撤，本章实验里哪种仓位更合适？若都不合适，你会怎么调整——降仓位，还是换策略？
# 

# %% Notebook cell 15 (code; source_chars=0)


# %% Notebook cell 16 (code; source_chars=0)


# %% [markdown] Notebook cell 17 (markdown)
# ## 本章总结
# 
# - **最大回撤** 衡量「最深的那次坑」；比最终赚多少更能反映真实持仓体验。
# - **水下曲线** 把回撤可视化——同时看 **深度** 与 **在水下的时间**。
# - **仓位管理** 是独立于选股的第二杠杆：同样信号，不同仓位 → 不同命运。
# - 专业流程往往是：先设定 **可承受回撤**（例如 −15%），再反推合理仓位——而不是先满仓再祈祷。
# 
# **下一章预告**：把视野从 **单只股票** 扩展到 **多标的组合**——相关性、分散投资，看看「不要把鸡蛋放在一个篮子里」在数据里长什么样。
# 

