from typing import Callable, List, Dict
import inspect
from functools import wraps

_callback_list: Dict[str, List[Callable[[any], any]]] = {}

def indicate_callback(callback_name: str):
    global _callback_list
    _callback_list[callback_name] = []
    def decorate(func: callable):
        @wraps(func)
        async def decorator(*args, **kwargs):
            output = await func(*args, **kwargs)
            for callback in _callback_list[callback_name]:
                callback(args=args, kwargs=kwargs, output=output)
            return output
        return decorator
    return decorate

def add_callback(callback_name: str, callback: Callable):
    global _callback_list
    _callback_list[callback_name].append(callback)