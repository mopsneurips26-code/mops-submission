from importlib import resources

import numpy as np
import polars as pl
from loguru import logger


def load_affordances(file_name):
    """Load affordances from a parquet file."""
    with resources.files("mopscasa.resources").joinpath(file_name).open() as f:
        df = pl.read_parquet(f)
        logger.info(f"Loaded {file_name} with {df.height} entries.")
        logger.info(df.head())
        return df, df.select("affordances").to_series()


def get_critical_affordances(merged_aff_col, aff_val_counts):
    """Identify critical affordances based on frequency."""
    critical_affordances = []
    for row in merged_aff_col:
        max_count = -1
        critical_aff = -1
        for aff in row:
            count = aff_val_counts.filter(pl.col("affordances") == aff)["count"][0]
            if count > max_count:
                max_count = count
                critical_aff = aff

        if critical_aff != -1 and critical_aff not in critical_affordances:
            critical_affordances.append(critical_aff)
    logger.info(
        f"Identified {len(critical_affordances)} critical affordances from data."
    )
    logger.info(critical_affordances)

    # Add affordances until we have at least 24
    if len(critical_affordances) < 24:
        for row in aff_val_counts.iter_rows():
            aff = row[0]
            if aff not in critical_affordances:
                critical_affordances.append(aff)
            if len(critical_affordances) >= 24:
                break
    return critical_affordances


def filter_affordances(df, critical_affordances: list[str]):
    """Filter affordances in the dataframe based on critical affordances."""
    df = df.with_columns(
        pl.col("affordances")
        .map_elements(
            lambda affs: [aff for aff in affs if aff in critical_affordances],
            return_dtype=pl.List(pl.String),
        )
        .alias("affordances")
    )

    # create new polars column with bitmask of affordances
    critical_affordances_id_map = {
        aff: idx for idx, aff in enumerate(critical_affordances)
    }

    df = df.with_columns(
        pl.col("affordances")
        .map_elements(
            lambda affs: sum(1 << critical_affordances_id_map[aff] for aff in affs),
            return_dtype=pl.UInt32,
        )
        .alias("affordance_bitmask"),
    )

    logger.info(f"Filtered dataframe now has {df.height} entries.")
    logger.info(df.head())

    return df


def main() -> None:
    # Load data
    logger.info("Loading data...")
    class_df, aff_col = load_affordances("class_affordances.parquet")
    robocasa_df, robocasa_aff_col = load_affordances("robocasa_affordances.parquet")

    # Merge and analyze affordances
    logger.info("Merging and analyzing affordances...")
    merged_aff_col = aff_col.append(robocasa_aff_col)
    aff_arr = np.concatenate(merged_aff_col.to_list())
    aff_series = pl.Series("affordances", aff_arr)
    aff_val_counts = aff_series.value_counts().sort("count", descending=True)

    # Identify critical affordances
    logger.info("Identifying 24 critical affordances...")
    critical_affordances = get_critical_affordances(merged_aff_col, aff_val_counts)
    logger.info(f"Critical affordances: {critical_affordances}")
    logger.info(f"Number of critical affordances: {len(critical_affordances)}")

    critical_affordances_id_map = {
        aff: idx for idx, aff in enumerate(critical_affordances)
    }
    # save to json in resources
    with (
        resources.files("mopscasa.resources")
        .joinpath("critical_affordances.json")
        .open("w") as f
    ):
        import json

        json.dump(critical_affordances_id_map, f, indent=4)

    # Filter dataframes
    logger.info("Filtering dataframes...")
    class_df = filter_affordances(class_df, critical_affordances)
    robocasa_df = filter_affordances(robocasa_df, critical_affordances)

    len_aff = robocasa_df.select(pl.col("affordances").map_elements(len)).to_series()
    logger.info(len_aff.value_counts().sort("affordances", descending=False))

    # Update for Polars
    class_df = class_df.with_columns(pl.col("model_cat").str.to_lowercase())
    robocasa_df = robocasa_df.with_columns(pl.col("model_cat").str.to_lowercase())

    # Save reduced annotations
    logger.info("Saving reduced annotations...")
    class_df.write_parquet(
        resources.files("mopscasa.resources").joinpath("red_class_aff.parquet")
    )
    robocasa_df.write_parquet(
        resources.files("mopscasa.resources").joinpath("red_robocasa_aff.parquet")
    )

    print("Reduced affordance annotations saved.")


if __name__ == "__main__":
    main()
