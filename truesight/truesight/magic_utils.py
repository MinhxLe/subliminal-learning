from typing import List, Type, TypeVar

T = TypeVar("T")


def extract_instances(cls: Type, target_class: Type[T]) -> List[T]:
    instances: List[T] = []

    # Check attributes at class level
    for name, attr in vars(cls).items():
        # Skip special attkributes and functions
        if name.startswith("__") or callable(attr):
            continue

        # If attribute is an instance of target_class, add it
        if isinstance(attr, target_class):
            instances.append(attr)
        # If attribute is a class, recursively search it
        elif isinstance(attr, type):
            instances.extend(extract_instances(attr, target_class))

    # Check nested classes
    for name, attr in cls.__dict__.items():
        if isinstance(attr, type) and not name.startswith("__"):
            instances.extend(extract_instances(attr, target_class))

    return instances
