import math
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class CorpusSet(Dataset):
    # @param docs   dictionary type of data, docs[document name] -> document content
    # @param mode   train or test
    def __init__(self, tokenizer, docs, queryData, neg_doc=None, mode='train'):
        self.docs = docs
        # pair data : query index, doeumet name, label
        self.data = []
        data = queryData
        self.querys = []
        for i in range(len(data)):
            self.querys.append(data[i]['query_text'])
            if mode=='train':
                pos_doc_names = data[i]['pos_doc_ids']
                pos_data_size = len(pos_doc_names)
                for doc_name in pos_doc_names:
                    self.data.append((i, doc_name, 1))
                if neg_doc is not None:
                    neg_doc_names = neg_doc[i]['neg_doc_ids']
                    for j in range(pos_data_size):
                        self.data.append((i, neg_doc_names[j], 0))
            else:
                for key in docs:
                    self.data.append((i, key))
        self.len = len(self.data)
        self.mode = mode
        self.tokenizer = tokenizer
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        if self.mode=='train':
            qIdx, docName, label = self.data[index]
        else:
            qIdx, docName = self.data[index]
            label = None
        query = self.querys[qIdx]
        content = self.docs[docName]
        if type(content) == float and math.isnan(content):
            content = " "
        # if set return_tensors='pt', the output data size will become [1, n]
        # token_ids = self.tokenizer.encode(query, content, max_length=512)
        tokens = self.tokenizer(f'{query}</s>{content}', max_length=512, truncation=True)
        # split data and to tensor
        token_ids = torch.tensor(tokens['input_ids'])
        atten_mask = torch.tensor(tokens['attention_mask'])
        return token_ids, atten_mask, label

def collate_fn(batches):
    token_ids = [d[0] for d in batches]
    atten_masks = [d[1] for d in batches]
    if batches[0][2] is not None:
        labels = torch.tensor([d[2] for d in batches])
    else:
        labels = None
    # padding
    token_ids = pad_sequence(token_ids, batch_first=True)
    atten_masks = pad_sequence(atten_masks, batch_first=True)
    # atten_masks = torch.zeros(token_ids.shape, dtype=torch.int64)
    # atten_masks = atten_masks.masked_fill(token_ids != 1, 1)
    return (token_ids, atten_masks, labels)

# @brief
# @param doc    document csv data
# @param train  training csv data
def generate_negative_data(doc, train):
    doc_ids = doc.to_dict('list')['doc_id']
    neg_dic = {
        'query_id' : [], 
        'neg_doc_ids' : [], 
    }
    keys = train.keys()
    rows, _ = train.shape
    for i in range(rows):
        neg_dic['query_id'].append(train['query_id'][i])
        neg_docs = doc_ids.copy()
        # print(len(neg_docs))
        pos_list = train['pos_doc_ids'][i].split(' ')
        # print(len(pos_list))
        for pos_doc in pos_list:
            neg_docs.remove(pos_doc)
        # print(len(neg_docs))
        neg_dic['neg_doc_ids'].append(" ".join(neg_docs))
    
    df = pd.DataFrame(neg_dic)
    df.to_csv('./ntust-ir2020-homework6/negative_data.csv', index=False)

# pandas usage
# https://stackoverflow.com/questions/26716616/convert-a-pandas-dataframe-to-a-dictionary
def get_csv_data():
    # { 
    #   name : content
    # }
    docs = pd.read_csv('./ntust-ir2020-homework6/documents.csv', header=0)
    docs = dict(docs.values.tolist())
    # [
    #     { 
    #         query_id : str, 
    #         query_text : str, 
    #         bm25_top1000 : [], 
    #         bm25_top1000_scores : []
    #     },
    # ]
    test = pd.read_csv('./ntust-ir2020-homework6/test_queries.csv', header=0)
    test = test.to_dict('records')
    for i in range(len(test)):
        test[i]['bm25_top1000'] = test[i]['bm25_top1000'].split(' ')
        test[i]['bm25_top1000_scores'] = test[i]['bm25_top1000_scores'].split(' ')
    # [
    #     { 
    #         query_id : str, 
    #         query_text : str, 
    #         pos_doc_ids : [], 
    #         bm25_top1000 : [], 
    #         bm25_top1000_scores : []
    #     },
    # ]
    train = pd.read_csv('./ntust-ir2020-homework6/train_queries.csv', header=0)
    train = train.to_dict('records')
    for i in range(len(train)):
        train[i]['pos_doc_ids'] = train[i]['pos_doc_ids'].split(' ')
        train[i]['bm25_top1000'] = train[i]['bm25_top1000'].split(' ')
        train[i]['bm25_top1000_scores'] = train[i]['bm25_top1000_scores'].split(' ')
    # [
    #     {
    #         query_id : str,
    #         neg_doc_ids : []
    #     },
    # ]
    neg_train = pd.read_csv('./ntust-ir2020-homework6/negative_data.csv', header=0)
    neg_train = neg_train.to_dict('records')
    for i in range(len(neg_train)):
        neg_train[i]['neg_doc_ids'] = neg_train[i]['neg_doc_ids'].split(' ')
    return docs, test, train, neg_train

if __name__ == '__main__':
    docs, test, train, neg_train = get_csv_data()

    # generate_negative_data(docs, train)
        
    # from transformers import RobertaTokenizer
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # max_length = 0
    # for i, content in enumerate(docs['doc_text']):
    #     if type(content) == float and math.isnan(content):
    #         print("AAAAAAAAA")
    #         break
    #     tmp = len(tokenizer.encode(content))
    #     if tmp > max_length:
    #         max_length = tmp
    #     # print(tmp)
    # print(f'max length : {max_length}')

    # df = pd.DataFrame({'a': ['red', 'yellow', 'blue'], 'b': [0.5, 0.25, 0.125]})
    # print(df.values)
    # print(df.values.tolist())
    # print(dict(df.values.tolist()))

    # from torch.utils.data import DataLoader
    # docs_dict = dict(docs.values.tolist())

    # trainset = CorpusSet(tokenizer, docs_dict, train, neg_doc=neg_train, mode='train')
    # trainloader = DataLoader(trainset, batch_size=2, collate_fn=collate_fn)

    # data = next(iter(trainloader))
    # print(data)
