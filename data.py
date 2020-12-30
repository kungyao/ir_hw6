import math
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class CorpusSet(Dataset):
    # @param docs   dictionary type of data, docs[document name] -> document content
    # @param mode   train or test
    def __init__(self, data, mode='train'):
        # pair data : query index, doeumet name, label
        self.data = data
        self.len = len(self.data)
        self.mode = mode
    def __len__(self):
        return self.len
    def get_query_doc_name(self, index):
        qIdx = self.data[index]['query_index']
        qName = self.data[index]['query_id']
        docName = self.data[index]['doc_id']
        if len(docName) == 1:
            docName = docName[0]
        return qIdx, qName, docName
    def __getitem__(self, index):
        data = self.data[index]
        label = 0 if self.mode=='train' else None
        return data, label

def collate_fn(batches):
    batchSize = len(batches)
    choiceSize = len(batches[0][0]['input_ids'])
    token_ids = []
    token_type_ids = []
    attention_mask = []

    if batches[0][1] is not None:
        labels = torch.tensor([d[1] for d in batches])
    else:
        labels = None

    for b in batches:
        data = b[0]
        for item in data['input_ids']:
            token_ids.append(torch.tensor(item))
        for item in data['token_type_ids']:
            token_type_ids.append(torch.tensor(item))
        for item in data['attention_mask']:
            attention_mask.append(torch.tensor(item))

    token_ids = pad_sequence(token_ids, batch_first=True)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)

    token_ids = torch.reshape(token_ids, (batchSize, choiceSize, -1))
    token_type_ids = torch.reshape(token_type_ids, (batchSize, choiceSize, -1))
    attention_mask = torch.reshape(attention_mask, (batchSize, choiceSize, -1))

    return (token_ids, token_type_ids, attention_mask, labels)

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

def my_tokenize(query_token_ids, tokenizer, max_length, docs, doc_name, doc_token_ids_lsit):
    if doc_name not in doc_token_ids_lsit:
        content = docs[doc_name]
        if type(content) == float and math.isnan(content):
            content = " "
        docToken = tokenizer.tokenize(content)
        doc_token_ids = tokenizer.convert_tokens_to_ids(docToken)
        doc_token_ids_lsit[doc_name] = doc_token_ids
    else:
        doc_token_ids = doc_token_ids_lsit[doc_name]
    input_ids = query_token_ids + doc_token_ids
    # token_type_ids
    token_type_ids = [0 for token_id in query_token_ids]
    token_type_ids.extend(1 for token_id in doc_token_ids)
    # clamp to specific size
    if len(input_ids) >= max_length:
        input_ids = input_ids[:max_length-1]
        token_type_ids = token_type_ids[:max_length-1]
    # add seperate token
    input_ids.append(tokenizer.sep_token_id)
    token_type_ids.append(tokenizer.sep_token_id)
    attention_mask = [1 for token_id in input_ids]
    return input_ids, token_type_ids, attention_mask, doc_token_ids_lsit

# collect data 
def preprocess_df(docs, test, train, tokenizer, negSize=3, max_length=512):
    doc_token_ids_lsit = {}
    testNew = []
    if test is not None:
        print(f'process test data')
        for i in range(len(test)):
            queryToken = tokenizer.tokenize(test[i]['query_text'])
            query_token_ids = tokenizer.convert_tokens_to_ids(queryToken)
            query_token_ids.insert(0, tokenizer.cls_token_id)
            query_token_ids.append(tokenizer.sep_token_id)
            for tp in test[i]['bm25_top1000']:
                input_ids, token_type_ids, attention_mask, doc_token_ids_lsit = my_tokenize(
                    query_token_ids,
                    tokenizer,
                    max_length,
                    docs,
                    tp,
                    doc_token_ids_lsit
                )
                subTest = {
                    'query_index'       : i,
                    'query_id'          : test[i]['query_id'],
                    'doc_id'            : [tp],
                    'input_ids'         : [torch.tensor(input_ids)],
                    'token_type_ids'    : [torch.tensor(token_type_ids)],
                    'attention_mask'    : [torch.tensor(attention_mask)],
                }
                testNew.append(subTest)
        print(f'process test data finish')
    trainNew = []
    if train is not None:
        print(f'process train data')
        for i in range(len(train)):
            queryToken = tokenizer.tokenize(train[i]['query_text'])
            query_token_ids = tokenizer.convert_tokens_to_ids(queryToken)
            query_token_ids.insert(0, tokenizer.cls_token_id)
            query_token_ids.append(tokenizer.sep_token_id)
            neg_doc_names = train[i]['neg_doc_ids']
            offset = 0
            for tp in train[i]['bm25_top1000']:
                input_ids, token_type_ids, attention_mask, doc_token_ids_lsit = my_tokenize(
                    query_token_ids,
                    tokenizer,
                    max_length,
                    docs,
                    tp,
                    doc_token_ids_lsit
                )
                subTrain = {
                    'query_index'       : i,
                    'query_id'          : train[i]['query_id'],
                    'doc_id'            : [tp],
                    'input_ids'         : [torch.tensor(input_ids)],
                    'token_type_ids'    : [torch.tensor(token_type_ids)],
                    'attention_mask'    : [torch.tensor(attention_mask)],
                }
                for j in range(offset, offset+negSize):
                    name = neg_doc_names[j%len(neg_doc_names)]
                    input_ids, token_type_ids, attention_mask, doc_token_ids_lsit = my_tokenize(
                        query_token_ids,
                        tokenizer,
                        max_length,
                        docs,
                        name,
                        doc_token_ids_lsit
                    )
                    subTrain['doc_id'].append(name)
                    subTrain['input_ids'].append(torch.tensor(input_ids))
                    subTrain['token_type_ids'].append(torch.tensor(token_type_ids))
                    subTrain['attention_mask'].append(torch.tensor(attention_mask))
                offset += negSize
                if offset >= len(neg_doc_names):
                    offset %= len(neg_doc_names)
                trainNew.append(subTrain)
        print(f'process train data finish')
    return testNew, trainNew

if __name__ == '__main__':
    # df = pd.DataFrame({'a': ['red', 'yellow', 'blue'], 'b': [0.5, 0.25, 0.125]})
    # print(df.values)
    # print(df.values.tolist())
    # print(dict(df.values.tolist()))

    from torch.utils.data import DataLoader
    from model import get_bert_model_and_tokenizer
    model, tokenizer = get_bert_model_and_tokenizer(False)

    docs, test, train = get_csv_data()
    newTest, newTrain = preprocess_df(docs, test, train, tokenizer)

    trainset = CorpusSet(newTrain, mode='train')
    trainloader = DataLoader(trainset, batch_size=1, collate_fn=collate_fn)

    for i, data in enumerate(trainloader):
        break

    data = next(iter(trainloader))
    print(data)
    print(data[0].shape)
