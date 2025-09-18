from _shared import *

# from dotenv import load_dotenv
# load_dotenv()


from transformers import AutoModel

import os
print(os.getenv('hf_endpoint'))

model = AutoModel.from_pretrained("albert-base-v2")
