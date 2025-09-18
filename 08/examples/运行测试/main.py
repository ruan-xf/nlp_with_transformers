import time
from transformers import pipeline
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

sample_text = '''(CNN) -- Usain Bolt rounded off the world championships Sunday by claiming his
third gold in Moscow as he anchored Jamaica to victory in the men's 4x100m
relay. The fastest man in the world charged clear of United States rival Justin
Gatlin as the Jamaican quartet of Nesta Carter, Kemar Bailey-Cole, Nickel
Ashmeade and Bolt won in 37.36 seconds. The U.S finished second in 37.56 seconds
with Canada taking the bronze after Britain were disqualified for a faulty
handover. The 26-year-old Bolt has n
'''

# 测试GPT-2-xl模型
@timeout(1800)
def test_gpt2():
    print("Testing GPT-2-xl...")
    start_time = time.time()
    pipe = pipeline("text-generation", model="gpt2-xl")
    result = pipe(sample_text+'\nTL;DR:')
    end_time = time.time()
    print(f"GPT-2-xl result: {result}")
    print(f"GPT-2-xl time: {end_time - start_time:.2f} seconds")
    return result, end_time - start_time

# 测试T5-large模型
@timeout(1800)
def test_t5():
    print("Testing T5-large...")
    start_time = time.time()
    pipe = pipeline("summarization", model="t5-large")
    result = pipe(sample_text)
    end_time = time.time()
    print(f"T5-large result: {result}")
    print(f"T5-large time: {end_time - start_time:.2f} seconds")
    return result, end_time - start_time

# 测试BART-large-cnn模型
@timeout(1800)
def test_bart():
    print("Testing BART-large-cnn...")
    start_time = time.time()
    pipe = pipeline("summarization", model="facebook/bart-large-cnn")
    result = pipe(sample_text)
    end_time = time.time()
    print(f"BART-large-cnn result: {result}")
    print(f"BART-large-cnn time: {end_time - start_time:.2f} seconds")
    return result, end_time - start_time

# 测试Pegasus模型
@timeout(1800)
def test_pegasus():
    print("Testing Pegasus...")
    start_time = time.time()
    pipe = pipeline("summarization", model="google/pegasus-cnn_dailymail")
    result = pipe(sample_text)
    end_time = time.time()
    print(f"Pegasus result: {result}")
    print(f"Pegasus time: {end_time - start_time:.2f} seconds")
    return result, end_time - start_time

# 运行测试
results = {}

try:
    results['gpt2'] = test_gpt2()
except TimeoutError:
    print("GPT-2-xl timed out after 30 minutes")
except Exception as e:
    print(f"GPT-2-xl error: {e}")

try:
    results['t5'] = test_t5()
except TimeoutError:
    print("T5-large timed out after 30 minutes")
except Exception as e:
    print(f"T5-large error: {e}")

try:
    results['bart'] = test_bart()
except TimeoutError:
    print("BART-large-cnn timed out after 30 minutes")
except Exception as e:
    print(f"BART-large-cnn error: {e}")

try:
    results['pegasus'] = test_pegasus()
except TimeoutError:
    print("Pegasus timed out after 30 minutes")
except Exception as e:
    print(f"Pegasus error: {e}")

# 打印结果摘要
print("\n=== 测试结果摘要 ===")
for model, result in results.items():
    if result:
        output, duration = result
        print(f"{model}: {duration:.2f} seconds")
    else:
        print(f"{model}: 超时或出错")