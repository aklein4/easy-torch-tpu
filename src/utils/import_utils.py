
import importlib

import utils.constants as constants


def import_class(
    path: str,
    *args,
) -> type:
    """
    Dynamically imports a class from a given module path.

    Args:
        path (str): The full path to the class, e.g. 'module.submodule.ClassName'.
        *args: Additional module path components to prepend, e.g. [supersupermodule, supermodule, ...].
    """

    # split into module and class name
    if '.' not in path:
        raise ImportError(f'Invalid path: {path}. Must be in the format ...module.submodule.ClassName')
    dot = path.rfind('.')
    module_name = path[:dot]
    class_name = path[dot + 1:]
    
    # handle nested modules
    for a in args[::-1]:
        module_name = f"{a}.{module_name}"

    # import the module
    module = importlib.import_module(module_name)

    # import the class
    if hasattr(module, class_name):
        class_ = getattr(module, class_name)
    else:
        raise ImportError(f'Could not find class {class_name} in module {module_name}.')
    
    return class_


# convenience functions for specific class types
def import_model(path: str) -> type:
    return import_class(
        path,
        constants.MODEL_MODULE,
    )

def import_trainer(path: str) -> type:
    return import_class(
        path,
        constants.TRAINER_MODULE,
    )

def import_collator(path: str) -> type:
    return import_class(
        path,
        constants.COLLATOR_MODULE,
    )

def import_optimizer(path: str) -> type:
    return import_class(
        path,
        constants.OPTIMIZER_MODULE,
    )
