"""在 Spyder 中按 F5，从本地 Parquet 读取股票原始日线。"""

import re
from datetime import datetime
from pathlib import Path

import pandas as pd


# 修改这里的股票代码和日期，就能查看本地保存的其他行情。
DATA_DIR = Path(r"D:\CNN_research\data_raw")
STOCK_CODE = "601318.SH"
START_DATE = "20250620"
END_DATE = "20250704"
COLUMNS = ["ts_code", "trade_date", "open", "high", "low", "close"]


def load_daily_quotes(
    data_dir: Path, stock_code: str, start_date: str, end_date: str
) -> pd.DataFrame:
    start = datetime.strptime(start_date, "%Y%m%d").strftime("%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d").strftime("%Y%m%d")
    if start > end:
        raise ValueError("开始日期不能晚于结束日期。")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"找不到本地行情目录：{data_dir}")

    stock_code = stock_code.strip().upper()
    if not re.fullmatch(r"[0-9]{6}(?:\.(?:SH|SZ|BJ))?", stock_code):
        raise ValueError("请输入六位股票代码，例如 601318 或 601318.SH。")
    symbols = [stock_code] if "." in stock_code else [
        f"{stock_code}.{exchange}" for exchange in ("SH", "SZ", "BJ")
    ]

    files = sorted(
        path for path in data_dir.glob("daily_" + "[0-9]" * 8 + ".parquet")
        if start <= path.stem.removeprefix("daily_") <= end
    )
    if not files:
        raise FileNotFoundError(f"本地目录中没有 {start} 至 {end} 的日线文件。")

    daily_quotes = [
        pd.read_parquet(
            path, engine="pyarrow", columns=COLUMNS,
            filters=[("ts_code", "in", symbols)],
        )
        for path in files
    ]
    quotes = pd.concat(daily_quotes, ignore_index=True)
    if quotes.empty:
        raise ValueError(f"在 {start} 至 {end} 的本地文件中找不到股票 {stock_code}。")
    if quotes["ts_code"].nunique() > 1:
        raise ValueError(f"股票代码 {stock_code} 匹配了多个市场，请加上交易所后缀。")

    quotes["trade_date"] = pd.to_datetime(quotes["trade_date"], format="%Y%m%d")
    if quotes["trade_date"].duplicated().any():
        raise ValueError(f"{stock_code} 有重复的交易日期，请核对源文件。")
    return quotes.sort_values("trade_date", ignore_index=True)


if __name__ == "__main__":
    quotes = load_daily_quotes(DATA_DIR, STOCK_CODE, START_DATE, END_DATE)
    print(f"本地原始日线：{STOCK_CODE}，共 {len(quotes)} 条。")
    print(quotes.to_string(index=False))
