import argparse
import csv
import re
from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql import functions as F


def create_spark_session(app_name: str = "MaldivesTradeProcessing") -> SparkSession:
    return SparkSession.builder.appName(app_name).getOrCreate()


def _normalize_header_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.strip().lower())


HEADER_NORMALIZATION = {
    "hscode": "hs_code",
    "code": "hs_code",
    "hsdescription": "description",
    "description": "description",
    "countryofconsignment": "country",
    "countryofdestination": "country",
    "country": "country",
    "unit": "unit",
    "quantity": "quantity",
    "quantitykg": "quantity",
    "cifmvr": "cif_mvr",
    "fobmvr": "cif_mvr",
    "rate": "rate",
}


def _find_header(file_path: Path) -> tuple[int, list[str]]:
    with file_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            try:
                tokens = next(csv.reader([line]))
            except StopIteration:
                continue
            normalized = [_normalize_header_token(token) for token in tokens if token.strip()]
            has_code = any(token in {"hscode", "code"} for token in normalized)
            has_quantity = "quantity" in normalized
            has_value = any(token in {"cifmvr", "fobmvr"} for token in normalized)
            if has_code and (has_quantity or has_value):
                return index, tokens
    raise ValueError(f"Missing header in {file_path.name}")


def _parse_line(line: str, headers: list[str]) -> Row | None:
    try:
        values = next(csv.reader([line], skipinitialspace=True))
    except StopIteration:
        return None
    if not values or len(values) < len(headers):
        return None
    clean_values = [value.strip() for value in values[: len(headers)]]
    return Row(**dict(zip(headers, clean_values)))


def _extract_year(file_path: Path) -> int:
    match = re.search(r"(\d{4})", file_path.name)
    if not match:
        raise ValueError(f"Year not found in {file_path.name}")
    return int(match.group(1))


def _numeric_column(col_name: str) -> F.Column:
    trimmed = F.trim(F.col(col_name))
    stripped = F.regexp_replace(trimmed, '"', "")
    stripped = F.regexp_replace(stripped, ",", "")
    return stripped.cast("double")


def _rename_columns(df: DataFrame) -> DataFrame:
    for column in df.columns:
        normalized = _normalize_header_token(column)
        target = HEADER_NORMALIZATION.get(normalized)
        if target and column != target:
            df = df.withColumnRenamed(column, target)
    return df


CANONICAL_COLUMNS = ["hs_code", "description", "country", "unit", "quantity", "cif_mvr", "rate"]


def _clean_dataframe(df: DataFrame, year: int, direction: str) -> DataFrame:
    df = _rename_columns(df)
    available_columns = [col for col in CANONICAL_COLUMNS if col in df.columns]
    df = df.select(*[F.col(col) for col in available_columns])
    numeric_targets = {"quantity", "cif_mvr", "rate"} & set(df.columns)
    for column in numeric_targets:
        df = df.withColumn(column, _numeric_column(column))
    if "country" in df.columns:
        df = df.withColumn("country", F.trim(F.col("country")))
        df = df.filter(F.length(F.coalesce(F.col("country"), F.lit(""))) > 0)
    if "hs_code" in df.columns:
        df = df.withColumn("hs_code", F.trim(F.regexp_replace(F.col("hs_code"), '"', "")))
        df = df.filter(F.length(F.coalesce(F.col("hs_code"), F.lit(""))) > 0)
    return df.withColumn("year", F.lit(year)).withColumn("direction", F.lit(direction))


def load_trade_file(spark: SparkSession, file_path: Path, direction: str) -> DataFrame:
    header_offset, headers = _find_header(file_path)
    headers = [value.strip() for value in headers if value.strip()]
    lines = (
        spark.sparkContext.textFile(str(file_path))
        .zipWithIndex()
        .filter(lambda pair: pair[1] > header_offset)
        .map(lambda pair: pair[0])
        .filter(lambda content: content.strip())
    )
    rows = (
        lines.map(lambda line: _parse_line(line, headers))
        .filter(lambda parsed: parsed is not None)
    )
    cleaned = spark.createDataFrame(rows)
    year = _extract_year(file_path)
    return _clean_dataframe(cleaned, year, direction)


def collect_trade_data(spark: SparkSession, data_dir: Path) -> DataFrame:
    files = sorted(data_dir.glob("*.csv"))
    frames = []
    for file_path in files:
        direction = (
            "import"
            if "Import" in file_path.name
            else "export"
            if "Export" in file_path.name
            else "unknown"
        )
        frames.append(load_trade_file(spark, file_path, direction))
    return reduce(
        lambda left, right: left.unionByName(right, allowMissingColumns=True),
        frames,
    )


def yearly_trade_summary(df: DataFrame) -> DataFrame:
    return (
        df.filter(F.col("cif_mvr").isNotNull())
        .groupBy("year", "direction")
        .agg(
            F.sum("cif_mvr").alias("total_cif_mvr"),
            F.sum("quantity").alias("total_quantity"),
            F.count("*").alias("line_count"),
        )
        .orderBy("year", "direction")
    )


def top_countries_by_value(df: DataFrame, top_n: int = 10, direction: str | None = None) -> DataFrame:
    subset = df.filter(F.col("cif_mvr").isNotNull())
    if direction:
        subset = subset.filter(F.col("direction") == direction)
    return (
        subset.groupBy("country")
        .agg(
            F.sum("cif_mvr").alias("total_cif"),
            F.sum("quantity").alias("total_quantity"),
        )
        .orderBy(F.desc("total_cif"))
        .limit(top_n)
    )


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_and_close(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cif_growth_by_direction(df: DataFrame, output_dir: Path) -> None:
    summary_pdf = yearly_trade_summary(df).toPandas()
    if summary_pdf.empty:
        return
    pivot = summary_pdf.pivot(index="year", columns="direction", values="total_cif_mvr").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(marker="o", ax=ax)
    ax.set_title("CIF Growth by Direction")
    ax.set_ylabel("Total CIF (MVR)")
    ax.set_xlabel("Year")
    ax.legend(title="Direction")
    _save_and_close(fig, output_dir / "cif_growth_by_direction.png")


def plot_top_trading_partners_by_year(df: DataFrame, output_dir: Path, top_n: int) -> None:
    base_summary = (
        df.filter(F.col("cif_mvr").isNotNull())
        .groupBy("year", "country")
        .agg(F.sum("cif_mvr").alias("total_cif"))
    )
    top_countries = (
        base_summary.groupBy("country")
        .agg(F.sum("total_cif").alias("market_cif"))
        .orderBy(F.desc("market_cif"))
        .limit(top_n)
    )
    country_list = [row["country"] for row in top_countries.collect()]
    if not country_list:
        return
    filtered = base_summary.filter(F.col("country").isin(country_list))
    pdf = filtered.toPandas()
    if pdf.empty:
        return
    pivot = pdf.pivot(index="year", columns="country", values="total_cif").fillna(0)
    fig, ax = plt.subplots(figsize=(14, 7))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(f"Top {len(country_list)} Trading Partners by Year")
    ax.set_ylabel("Total CIF (MVR)")
    ax.set_xlabel("Year")
    ax.legend(title="Country", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_and_close(fig, output_dir / "top_trading_partners_by_year.png")


def plot_top_hs_chapters(df: DataFrame, output_dir: Path, top_n: int) -> None:
    chapters = (
        df.filter(F.col("cif_mvr").isNotNull() & F.col("hs_code").isNotNull())
        .withColumn("hs_chapter", F.substring(F.col("hs_code"), 1, 2))
        .filter(F.col("hs_chapter").rlike(r"^[0-9]+$"))
        .groupBy("hs_chapter")
        .agg(F.sum("cif_mvr").alias("total_cif"))
        .orderBy(F.desc("total_cif"))
        .limit(top_n)
    )
    pdf = chapters.toPandas()
    if pdf.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(pdf["hs_chapter"], pdf["total_cif"])
    ax.set_title(f"Top {len(pdf)} HS Chapters by CIF")
    ax.set_xlabel("HS Chapter")
    ax.set_ylabel("Total CIF (MVR)")
    _save_and_close(fig, output_dir / "top_hs_chapters.png")


def run_analysis(data_dir: Path, top_n: int, charts_dir: Path) -> None:
    spark = create_spark_session()
    try:
        trade_df = collect_trade_data(spark, data_dir)
        trade_df.cache()
        summary = yearly_trade_summary(trade_df)
        print("\n=== Yearly CIF summary ===")
        summary.show(truncate=False)
        print("\n=== Top importing countries ===")
        top_countries_by_value(trade_df, top_n=top_n, direction="import").show(truncate=False)
        print("\n=== Top exporting countries ===")
        top_countries_by_value(trade_df, top_n=top_n, direction="export").show(truncate=False)
        _ensure_directory(charts_dir)
        plot_cif_growth_by_direction(trade_df, charts_dir)
        plot_top_trading_partners_by_year(trade_df, charts_dir, top_n)
        plot_top_hs_chapters(trade_df, charts_dir, top_n)
        print(f"\nCharts written to {charts_dir.resolve()}")
    finally:
        spark.stop()


def main() -> None:
    plt.rcParams.update({'font.size': 14})
    parser = argparse.ArgumentParser(
        description="Preprocess Maldives import/export data with PySpark."
    )
    parser.add_argument("--data-dir", default="data", help="Directory storing the CSV exports.")
    parser.add_argument("--top-n", type=int, default=10, help="How many top countries to show.")
    parser.add_argument("--charts-dir", default="charts", help="Directory where charts are saved.")
    args = parser.parse_args()
    run_analysis(Path(args.data_dir), args.top_n, Path(args.charts_dir))


if __name__ == "__main__":
    main()
