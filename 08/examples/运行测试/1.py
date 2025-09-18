from transformers import pipeline

prompt = "Hugging Face is a community-based open-source platform for machine learning."
# 默认使用 'openai-community/gpt2'
generator = pipeline(task="text-generation")
generator(prompt)  # doctest: +SKIP