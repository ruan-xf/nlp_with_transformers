
import sys, os
# 交互窗口无需进行此设置，终端中也应加载此.env设置的，目前只能进行此处理
from dotenv import load_dotenv
load_dotenv()
print(os.getenv('hf_endpoint'))

sys.path.append("..")

from common import *

