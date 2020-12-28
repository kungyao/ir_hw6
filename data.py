import math
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class CorpusSet(Dataset):
    # @param docs   dictionary type of data, docs[document name] -> document content
    # @param mode   train or test
    def __init__(self, tokenizer, docs, queryData, negSize=3, mode='train'):
        self.docs = docs
        # pair data : query index, doeumet name, label
        self.data = []
        data = queryData
        self.querys = []
        self.negSize = negSize
        for i in range(len(data)):
            self.querys.append((data[i]['query_id'], data[i]['query_text']))
            if mode=='train':
                pos_doc_names = data[i]['pos_doc_ids']
                offset = 0
                neg_doc_names = data[i]['neg_doc_ids']
                neg_data_size = len(neg_doc_names)
                for doc_name in pos_doc_names:
                    # query index, pos doc, neg doc * n
                    subdata = [i, doc_name]
                    for j in range(offset, offset + negSize):
                        subdata.append(neg_doc_names[j%neg_data_size])
                    offset += negSize
                    offset %= neg_data_size
                    self.data.append(subdata)
            else:
                for doc_name in data[i]['bm25_top1000']:
                    self.data.append([i, doc_name])
        self.len = len(self.data)
        self.mode = mode
        self.tokenizer = tokenizer
    def __len__(self):
        return self.len
    def get_query_doc_name(self, index):
        qIdx, docName = self.data[index][:2]
        return qIdx, self.querys[qIdx][0], docName
    def __getitem__(self, index):
        data = self.data[index]
        qIdx = data[0]
        docs = data[1:]
        if self.mode=='train':
            label = 0
        else:
            label = None
        query = self.querys[qIdx][1]
        # query = "TEST ABC"
        tokenPair = []
        # i = 0
        for doc in docs:
            tmp = self.docs[doc]
            if type(tmp) == float and math.isnan(tmp):
                tmp = " "
            # tmp = ''.join([f'TEST_{i}'] * i * (index + 1))
            # i += 1
            tokenPair.append(tokenizer(query, tmp, max_length=512, padding=True))
        return tokenPair, label

def collate_fn(batches):
    batchSize = len(batches)
    choiceSize = len(batches[0][0])
    token_ids = []
    token_type_ids = []
    attention_mask = []

    if batches[0][1] is not None:
        labels = torch.tensor([d[1] for d in batches])
    else:
        labels = None

    for b in batches:
        for item in b[0]:
            token_ids.append(torch.tensor(item['input_ids']))
            token_type_ids.append(torch.tensor(item['token_type_ids']))
            attention_mask.append(torch.tensor(item['attention_mask']))

    token_ids = pad_sequence(token_ids, batch_first=True)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)

    token_ids = torch.reshape(token_ids, (batchSize, choiceSize, -1))
    token_type_ids = torch.reshape(token_type_ids, (batchSize, choiceSize, -1))
    attention_mask = torch.reshape(attention_mask, (batchSize, choiceSize, -1))

    return (token_ids, token_type_ids, attention_mask, labels)

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

    from torch.utils.data import DataLoader
    from model import get_bert_model_and_tokenizer
    model, tokenizer = get_bert_model_and_tokenizer(False)

    trainset = CorpusSet(tokenizer, docs, train, mode='train')
    trainloader = DataLoader(trainset, batch_size=2, collate_fn=collate_fn)

    data = next(iter(trainloader))
    print(data)
    print(data[0].shape)

    # from model import get_bert_model_and_tokenizer
    # model, tokenizer = get_bert_model_and_tokenizer(False)

    # print(tokenizer.pad_token_id)
    # print(tokenizer.cls_token_id)
    # print(tokenizer.sep_token_id)

    # prompt = "TEST ABC"
    # choice0 = "TEST_0"
    # choice1 = "TEST_1"
    # choice2 = "TEST_2"
    # choice3 = "TEST_3"
    # # encoding = tokenizer(prompt, choice0, padding=True)
    # # encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='pt', padding=True)
    # # print(encoding)
    # print(tokenizer(prompt, choice0, padding=True))
    # print(tokenizer(prompt, choice1, padding=True))
    # print(tokenizer(prompt, choice2, padding=True))
    # print(tokenizer(prompt, choice3, padding=True))