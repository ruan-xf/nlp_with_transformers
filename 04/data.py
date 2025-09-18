import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# df_train, df_test, df_val = (
#     pd.read_csv('data/'+fn) for fn in 
#     ('train.csv', 'test.csv', 'validation.csv')
# )

# [len(df) for df in (df_train, df_test, df_val)]
# [3668, 1725, 408]

class CustomDataset(Dataset):

    def __init__(self, df_type: str, maxlen, bert_model):

        self.data = pd.read_csv(f'data/{df_type}.csv')
        #Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)  

        self.maxlen = maxlen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        #根据索引索取DataFrame中句子1余句子2
        sent1 = str(self.data.loc[index, 'sentence1'])
        sent2 = str(self.data.loc[index, 'sentence2'])

        # 对句子对分词，得到input_ids、attention_mask和token_type_ids
        encoded_pair = self.tokenizer(sent1, sent2, 
                                      padding='max_length',  # 填充到最大长度
                                      truncation=True,  # 根据最大长度进行截断
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # 返回torch.Tensor张量
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # padded values对应为 "0" ，其他token为1
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  #第一个句子的值为0，第二个句子的值为1 # 只有一句全为0

        label = self.data.loc[index, 'label']
        return token_ids, attn_masks, token_type_ids, label  



# ds = CustomDataset('validation')
# batch_size = 4

# loader = DataLoader(ds, batch_size)
# batch = next(iter(loader))
# batch


# batch: list
# [t.shape for t in batch]
# [torch.Size([4, 128]),
#  torch.Size([4, 128]),
#  torch.Size([4, 128]),
#  torch.Size([4])]


class SentencePairDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()