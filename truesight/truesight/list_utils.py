from typing import Generator, Iterable, TypeVar, List, Callable, Any, Dict


def split(iterable, condition_fn):
    group_true = []
    group_false = []

    for item in iterable:
        if condition_fn(item):
            group_true.append(item)
        else:
            group_false.append(item)

    return (group_true, group_false)


T = TypeVar("T")  # Generic type for the elements in the iterable


def sort_by_value_order(
    iterable: List[T], value_order: List[Any], key: Callable[[T], Any] = lambda x: x
) -> List[T]:
    """
    Sort a list based on a specified value order.

    Args:
        iterable: The list to be sorted
        value_order: List defining the desired order of values
        key: Function to extract the value to sort by from each element

    Returns:
        A new sorted list
    """
    # Create a mapping from values to their positions in value_order
    order_dict: Dict[Any, int] = {val: idx for idx, val in enumerate(value_order)}

    # Define a custom sorting function
    def get_sort_key(item: T) -> int:
        item_value = key(item)
        # Return the position in value_order if present, otherwise a large number to put it at the end
        return order_dict.get(item_value, len(order_dict))

    # Return the sorted list
    return sorted(iterable, key=get_sort_key)


def batch(iterable: Iterable[T], batch_size: int) -> Generator[List[T], None, None]:
    """
    Process an iterable in batches of specified size.

    Args:
        iterable: The source iterable to be batched
        batch_size: The size of each batch

    Returns:
        A generator yielding lists of items, each list containing up to batch_size items

    Raises:
        ValueError: If batch_size is less than or equal to 0
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    current_batch = []

    for item in iterable:
        current_batch.append(item)

        if len(current_batch) >= batch_size:
            yield current_batch
            current_batch = []

    if current_batch:
        yield current_batch


def flatten(lists: List[List[T]]) -> List[T]:
    all = []
    for list_ in lists:
        all.extend(list_)
    return all


def flatten_recursive(nested_list) -> List:
    """
    Recursively flatten a nested list structure of arbitrary depth.

    Args:
        nested_list: A nested list structure that may contain lists at multiple levels

    Returns:
        A flattened list containing all non-list elements from the nested structure
    """
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_recursive(item))
        else:
            result.append(item)
    return result
