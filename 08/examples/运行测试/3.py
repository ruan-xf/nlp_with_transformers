from huggingface_hub import snapshot_download

repos = '''\
gpt2-xl
t5-large
facebook/bart-large-cnn
google/pegasus-cnn_dailymail
transformersbook/pegasus-samsum\
'''
for repo in repos.splitlines():
    snapshot_download(repo)