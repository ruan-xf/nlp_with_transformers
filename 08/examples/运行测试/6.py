import time
from functools import wraps
import signal

# 超时处理装饰器
class TimeoutError(Exception):
    pass

def timeout(seconds=1800, error_message="Function call timed out"):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator


@timeout(30)
def sleep_50():
    print('test starts')
    time.sleep(50)
    print('slept 50s, failed!')


sleep_50()