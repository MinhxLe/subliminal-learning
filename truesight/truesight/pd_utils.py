from loguru import logger
import pandas as pd


def sort_by_value_order(
    df: pd.DataFrame,
    value_col: str,
    value_order,
) -> pd.DataFrame:
    # Create mapping for known values
    value_order_dict = {v: i for i, v in enumerate(value_order)}

    # Find values not in the mapping
    unknown_values = set(df[value_col].unique()) - set(value_order)

    # Print warning for unknown values
    if unknown_values:
        logger.warning(f"{len(unknown_values)} values were not found")

    # Custom sorting function that puts unknown values at the end
    def custom_sort(series):
        max_order = len(value_order)
        return series.map(lambda x: value_order_dict.get(x, max_order))

    return df.sort_values(value_col, key=custom_sort, ascending=True)
