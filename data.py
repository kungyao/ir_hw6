import math
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class CorpusSet(Dataset):
    # @param docs   dictionary type of data, docs[document name] -> document content
    # @param mode   train or test
    def __init__(self, tokenizer, docs, queryData, mode='train'):
        self.docs = docs
        # pair data : query index, doeumet name, label
        self.data = []
        data = queryData
        self.querys = []
        for i in range(len(data)):
            self.querys.append((data[i]['query_id'], data[i]['query_text']))
            if mode=='train':
                pos_doc_names = data[i]['pos_doc_ids']
                for doc_name in pos_doc_names:
                    self.data.append((i, doc_name, 1))
                neg_data_size = len(pos_doc_names) * 3
                neg_doc_names = data[i]['neg_doc_ids']
                if neg_data_size > len(neg_doc_names):
                    neg_data_size = len(neg_doc_names)
                for j in range(neg_data_size):
                    self.data.append((i, neg_doc_names[j], 0))
            else:
                for doc_name in data[i]['bm25_top1000']:
                    self.data.append((i, doc_name))
        self.len = len(self.data)
        self.mode = mode
        self.tokenizer = tokenizer
    def __len__(self):
        return self.len
    def get_query_doc_name(self, index):
        qIdx, docName = self.data[index][:2]
        return qIdx, self.querys[qIdx][0], docName
    def __getitem__(self, index):
        if self.mode=='train':
            qIdx, docName, label = self.data[index]
        else:
            qIdx, docName = self.data[index]
            label = None
        query = self.querys[qIdx][1]
        content = self.docs[docName]
        if type(content) == float and math.isnan(content):
            content = " "

        word_pieces = ['[CLS]']
        qtoken = self.tokenizer.tokenize(query)
        word_pieces += qtoken + ["[SEP]"]
        len_a = len(word_pieces)

        dtoken = self.tokenizer.tokenize(content)
        word_pieces += dtoken
        if len(word_pieces) >= 512:
            word_pieces = word_pieces[:511]
        word_pieces += ["[SEP]"]
        len_b = len(word_pieces) - len_a

        token_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(word_pieces))
        token_type_ids = torch.tensor([0] * len_a + [1] * len_b, dtype=torch.int64)
        return token_ids, token_type_ids, label

def collate_fn(batches):
    token_ids = [d[0] for d in batches]
    token_type_ids = [d[1] for d in batches]
    if batches[0][2] is not None:
        labels = torch.tensor([d[2] for d in batches])
    else:
        labels = None
    # padding
    token_ids = pad_sequence(token_ids, batch_first=True)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True)
    # gen attention mask
    atten_masks = torch.zeros(token_ids.shape, dtype=torch.int64)
    atten_masks = atten_masks.masked_fill(token_ids != 0, 1)
    return (token_ids, token_type_ids, atten_masks, labels)

# @brief generate negative from all document that do not in positive doc
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
    #         bm25_top1000 : [str], 
    #         bm25_top1000_scores : [float]
    #     },
    # ]
    test = pd.read_csv('./ntust-ir2020-homework6/test_queries.csv', header=0)
    test = test.to_dict('records')
    for i in range(len(test)):
        test[i]['bm25_top1000'] = test[i]['bm25_top1000'].split(' ')
        tmpList = test[i]['bm25_top1000_scores'].split(' ')
        test[i]['bm25_top1000_scores'] = [float(s) for s in tmpList]
    # [
    #     { 
    #         query_id : str, 
    #         query_text : str, 
    #         pos_doc_ids : [str], 
    #         bm25_top1000 : [str], 
    #         bm25_top1000_scores : [float]
    #     },
    # ]
    train = pd.read_csv('./ntust-ir2020-homework6/train_queries.csv', header=0)
    train = train.to_dict('records')
    for i in range(len(train)):
        train[i]['pos_doc_ids'] = train[i]['pos_doc_ids'].split(' ')
        train[i]['bm25_top1000'] = train[i]['bm25_top1000'].split(' ')
        tmpList = train[i]['bm25_top1000_scores'].split(' ')
        train[i]['bm25_top1000_scores'] = [float(s) for s in tmpList]
        train[i]['neg_doc_ids'] = [name for name in train[i]['bm25_top1000'] if name not in train[i]['pos_doc_ids']]
    return docs, test, train

if __name__ == '__main__':
    docs, test, train = get_csv_data()

    # generate_negative_data(docs, train)
    
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

    print('train---------------------')
    for i in range(len(train)):
        for pos_id in train[i]['pos_doc_ids']:
            if pos_id not in train[i]['bm25_top1000']:
                print(train[i]['query_id'], pos_id)
                break