
import sys, os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv('hf_endpoint'))

sys.path.append("..")

from common import *

