import argparse
import pandas as pd

import torch
from torch.utils.data import DataLoader

from data import CorpusSet, collate_fn, get_csv_data
from model import get_albert_model_and_tokenizer
from utils import mean_average_precision

# @param resultList dict, element wieth type [document name:str] -> score:float
# @param testDict   
def calculate_query_mAP(resultList, testDict, outPath):
    assert len(resultList) == len(testDict)
    mAP = 0
    output = {
        'query_id' : [], 
        'ranked_doc_ids' : [], 
    }
    for i in range(len(resultList)):
        doc = resultList[i]
        test = testDict[i]
        for j in range(topx):
            doc[test['bm25_top1000'][j]] += alpha * test['bm25_top1000_scores'][j]
        # dict to lsit
        doc = [(key, doc[key]) for key in doc]
        # sort
        doc = sorted(doc, key = lambda s: s[1], reverse = True)
        # clamp
        # doc = doc[:topx]
        doc = [doc[j][0] for j in range(topx)]
        mAP += mean_average_precision(doc, test['pos_doc_ids'])

        output['query_id'].append(testDict['query_id'])
        output['ranked_doc_ids'].append(' '.join(doc))
    
    mAP /= len(resultList)
    print(f'mAP : {mAP}')
    # to csv
    df = pd.DataFrame(output)
    df.to_csv(outPath, index=False)

def predict_from_model(args, docs, test, train):
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    
    model, tokenizer = get_albert_model_and_tokenizer(ifModel=False)
    model = torch.load(args.model)
    model.to(device)
    
    topx = 1000
    thres = topx - 1
    testset = CorpusSet(tokenizer, docs, train, mode='test')
    testloader = DataLoader(testset, batch_size=1, collate_fn=collate_fn)

    rank = [0] * topx
    resultList = {
        'query_id'      : [], 
        'doc_ids'       : [], 
        'doc_scores'    : [], 
    }
    with torch.no_grad():
        for i, data in enumerate(testloader):
            token_ids = data[0].to(device)
            res = model(input_ids=token_ids)
            rank[i % topx] = res.logits[0][1].cpu().item()
            if i != 0 and i % thres == 0:
                ind, qName, _ = testset.get_query_doc_name(i)
                rankStr = [str(r) for r in rank]
                resultList['query_id'].append(qName)
                resultList['doc_ids'].append(' '.join(train[ind]['bm25_top1000']))
                resultList['doc_scores'].append(' '.join(rankStr))
                rank = [0] * topx
    df = pd.DataFrame(resultList)
    df.to_csv('./ntust-ir2020-homework6/bert_result.csv', index=False)

def get_args():
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-mdl', '--model', default=None, dest='model', help='./models/epoch_0.pth')
    # parser.add_argument('-bs', '--batch_size', type=int, default=1, dest='batch_size')
    parser.add_argument('-test', '--test', type=str, required=True, dest='test')
    parser.add_argument('-map', '--map', action='store_true')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    docs, test, train = get_csv_data()
    if args.model:
        predict_from_model(args, docs, test, train)

    # alpha = 5
    # data = train if args.test == 'train' else test
    # br = pd.read_csv('./ntust-ir2020-homework6/bert_result.csv', header=0)
    # br = br.to_dict('records')
    # for i in range(len(br)):
    #     res = br[i]
    #     gt = data[i]


    if args.map and args.test == 'train':
        # calculate_query_mAP(br, train, )
        pass
