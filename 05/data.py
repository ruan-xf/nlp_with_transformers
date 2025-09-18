import os
import shutil
import pytorch_lightning as pl
import torch
from torch.utils.data import (
    random_split, DataLoader,
    TensorDataset
)
from transformers import AutoTokenizer, BertTokenizer
import csv
import pandas as pd
import numpy as np


class MyDefaultDataModule(pl.LightningDataModule):
    # 定义所有实体标签：B-{type}和I-{type}，排除特定类型
    entity_labels = [
        f'{prefix}-{entity_type}'
        for prefix in ['B', 'I'] 
        for entity_type in range(1, 54+1) 
        if entity_type not in (27, 45)  # 排除不需要的实体类型
    ]

    # 添加非实体标签'O'
    all_labels = entity_labels + ['O']
    label_id_to_name = pd.Series(all_labels)
    label_name_to_id = pd.Series(label_id_to_name.index, index=all_labels)


    def __init__(
            self,
            max_len=105,
            model_name = 'hfl/chinese-roberta-wwm-ext',
            batch_size=32,
            preprocessed_dir = 'data/preprocessed',
        ):
        super().__init__()
        self.save_hyperparameters()

        self.train_fn, self.val_fn, self.test_fn, self.predict_fn = [
            os.path.join(preprocessed_dir, f'{data_type}_data.pt')
            for data_type in ('train', 'val', 'test', 'predict')
        ]

        self.tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(model_name)


    def prepare_data(self):
        preprocessed_dir = self.hparams.preprocessed_dir
        if os.path.exists(preprocessed_dir):
            # shutil.rmtree(preprocessed_dir)
            # os.makedirs(preprocessed_dir)
            return


        os.makedirs(preprocessed_dir, exist_ok=True)
        data = pd.read_csv(
            'data/train.txt',
            sep=r'\s+',
            header=None,
            names = ['char', 'tag'],
            # nrows=200,
            skip_blank_lines=False,
            quoting=csv.QUOTE_NONE
        )
        # 先处理空行
        null_rows = data.isnull().all(axis=1)
        data['sentence_id'] = null_rows.cumsum()
        data = data[~null_rows]

        data.loc[pd.isna(data.tag), ['char', 'tag']] = ['[SEP]', 'O']
        
        label_name_to_id = self.label_name_to_id

        data['label_id'] = label_name_to_id[data.tag].values

        grouped = data[['sentence_id', 'char', 'label_id']].groupby('sentence_id')


        data = grouped.apply(
            lambda x: pd.Series({
                'sentence': ' '.join(x.char),
                'labeling': x.reset_index(drop=True)
            }), include_groups=False
        )

        max_len = self.hparams.max_len

        # 对句子进行tokenization编码
        def batch_encode_plus(sentences: list[str]):
            return self.tokenizer.batch_encode_plus(
                sentences,
                add_special_tokens=True,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                return_attention_mask=True,
                return_token_type_ids=False,
            )

        encoded_sentences = batch_encode_plus(data.sentence.tolist())

        # 获取每个token对应的原始单词位置
        token_to_char_mapping = pd.Series(np.arange(len(data))).apply(encoded_sentences.word_ids)

        def assign_labels_to_tokens(original_labels, token_char_mapping):
            # 为特殊token添加默认标签
            original_labels.loc[-1] = (None, label_name_to_id['O'])
            
            # 创建token到标签的映射表
            token_labels = pd.DataFrame({
                'char_position': pd.Series(token_char_mapping).fillna(-1).astype(int),
                'label': None
            })
            
            # 将原始单词标签分配给对应的token
            token_labels['label'] = original_labels.loc[token_labels.char_position, 'label_id'].values
            
            return token_labels.label.tolist()

        # 为每个句子的token分配标签
        outputs = pd.concat([data.labeling, token_to_char_mapping], axis=1).apply(
            lambda row: assign_labels_to_tokens(*row),
            axis=1
        ).tolist()

        outputs = torch.LongTensor(outputs)
        encoded_sentences['outputs'] = outputs



        # do split: train, val, test
        assert list(encoded_sentences.keys()) == ['input_ids', 'attention_mask', 'outputs']
        dataset = TensorDataset(*encoded_sentences.values())

        for ds, fn in zip(
            random_split(dataset, (.8, .1, .1)),
            (self.train_fn, self.val_fn, self.test_fn)
        ):
            torch.save(ds, fn)

        # predict_data:
        with open(
            'data/sample_per_line_preliminary_A.txt',
        ) as f:
            sentences = pd.Series(
                # line for i, line in enumerate(f) if i < 200
                f
            )


        sentences = sentences.apply(
            lambda x: (
                pd.Series(list(x))
                .str.replace(r'\s+', '[SEP]', regex=True)
                .tolist()
            )
        ).str.join(' ').tolist()

        encoded_sentences = batch_encode_plus(sentences)
        
        dataset = TensorDataset(*encoded_sentences.values())
        torch.save(dataset, self.predict_fn)



    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = torch.load(self.train_fn, weights_only=False)

        if stage in ("fit", "validate"):
            self.val_dataset = torch.load(self.val_fn, weights_only=False)

        if stage == "test":
            self.test_dataset = torch.load(self.test_fn, weights_only=False)

        if stage == "predict":
            self.predict_dataset = torch.load(self.predict_fn, weights_only=False)
        
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size)
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size)

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.hparams.batch_size)

# dm = MyDefaultDataModule()
# dm.prepare_data()

# # splits/transforms
# dm.setup(stage="fit")
# # dm.setup('predict')

# # use data
# batch = next(iter(dm.train_dataloader()))
# # batch = next(iter(dm.predict_dataloader()))

# from transformers import BertForTokenClassification
# model = BertForTokenClassification.from_pretrained('hfl/chinese-roberta-wwm-ext', num_labels=len(MyDefaultDataModule.all_labels))

# out = model(batch[0], batch[1], labels=batch[2])
# # model(*batch)

