"""从本地 Parquet 读取原始 A 股日线，可按需导出 CSV 快照。"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory


def save_snapshot(quotes, data_dir: Path, output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在，请指定一个新目录：{output}")

    symbol = quotes["ts_code"].iat[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".local-quotes-", dir=output.parent) as temporary:
        snapshot = Path(temporary) / "snapshot"
        snapshot.mkdir()
        csv_path = snapshot / f"{symbol.split('.')[0]}_raw.csv"
        quotes.to_csv(csv_path, index=False, encoding="utf-8-sig")
        manifest = {
            "source": "local Parquet",
            "data_dir": str(data_dir.resolve()),
            "symbol": symbol,
            "price_basis": "raw",
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": {
                csv_path.name: {
                    "rows": len(quotes),
                    "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                },
            },
        }
        (snapshot / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        snapshot.rename(output)
    print(f"已保存本地行情快照：{output}")


def main():
    local_reader = run_path(str(Path(__file__).with_name("load_local_daily.py")))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=local_reader["DATA_DIR"], help="本地 Parquet 目录")
    parser.add_argument("--symbol", default=local_reader["STOCK_CODE"], help="六位股票代码，可带交易所后缀")
    parser.add_argument("--start", default=local_reader["START_DATE"], help="YYYYMMDD")
    parser.add_argument("--end", default=local_reader["END_DATE"], help="YYYYMMDD")
    parser.add_argument("--output", type=Path, help="可选的快照目录，须尚未创建")
    args = parser.parse_args()

    quotes = local_reader["load_daily_quotes"](args.data_dir, args.symbol, args.start, args.end)
    print(f"本地原始日线：{quotes['ts_code'].iat[0]}，共 {len(quotes)} 条。")
    print(quotes.to_string(index=False))
    if args.output is not None:
        save_snapshot(quotes, args.data_dir, args.output)
    return quotes


if __name__ == "__main__":
    quotes = main()
