import argparse
import pandas as pd

import torch
from torch.utils.data import DataLoader

from data import CorpusSet, collate_fn, get_csv_data
from model import get_bert_model_and_tokenizer
from utils import mean_average_precision

def prepare_bert_result(test_type):
    br = pd.read_csv(f'./ntust-ir2020-homework6/bert_result_{test_type}.csv', header=0)
    br = br.to_dict('records')
    for i in range(len(br)):
        br[i]['doc_ids'] = br[i]['doc_ids'].split(' ')
        tmpList = br[i]['doc_scores'].split(' ')
        br[i]['doc_scores'] = [float(s) for s in tmpList]
    return br
  
def get_final_result(args, docs, data, alpha=2):
    mAP = 0
    output = {
        'query_id' : [], 
        'ranked_doc_ids' : [], 
    }
    br = prepare_bert_result(args.test)
    for i in range(len(br)):
        res = br[i]
        test = data[i]
        rank = []
        for j in range(len(test['bm25_top1000'])):
            rank.append((
                res['doc_ids'][j],
                alpha * res['doc_scores'][j] + test['bm25_top1000_scores'][j]
            ))
        # sort
        rank = sorted(rank, key = lambda s: s[1], reverse = True)
        rank = [r[0] for r in rank]

        output['query_id'].append(test['query_id'])
        output['ranked_doc_ids'].append(' '.join(rank))
    
    # to csv
    df = pd.DataFrame(output)
    df.to_csv(f'./ntust-ir2020-homework6/result_{args.test}.csv', index=False)

def test_alpha_by_use_map(args, docs, data):
    br = prepare_bert_result(args.test)

    maxAlpha = 5 * 10 + 1
    fianlAlpha = 0
    maxmAP = 0
    mAPList = []
    for alpha in range(1, maxAlpha):
        alpha *= 0.1
        mAPSub = 0
        subList = []
        for i in range(len(br)):
            res = br[i]
            gt = data[i]
            #
            rank = []
            for j in range(len(res['doc_ids'])):
                rank.append((res['doc_ids'][j], alpha * res['doc_scores'][j] + gt['bm25_top1000_scores'][j]))
            rank = sorted(rank, key = lambda s: s[1], reverse = True)
            rank = [r[0] for r in rank]

            tmp = mean_average_precision(rank, gt['pos_doc_ids'])
            subList.append(tmp)
            mAPSub += tmp
        #
        if mAPSub > maxmAP:
            maxmAP = mAPSub
            fianlAlpha = alpha
            mAPList = subList
    print(f'average alpha : {fianlAlpha}, mAP : {maxmAP / len(br)}, mAPList : {mAPList}')
    return fianlAlpha

def predict_from_model(args, docs, data):
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    
    model, tokenizer = get_bert_model_and_tokenizer(ifModel=False)
    model = torch.load(args.model)
    model.to(device)
    
    topx = 1000
    testset = CorpusSet(tokenizer, docs, data, mode='test')
    testloader = DataLoader(testset, batch_size=1, collate_fn=collate_fn)

    rank = [0] * topx
    resultList = {
        'query_id'      : [], 
        'doc_ids'       : [], 
        'doc_scores'    : [], 
    }
    with torch.no_grad():
        for i, tt in enumerate(testloader):
            token_ids = tt[0].to(device)
            token_type_ids = tt[1].to(device)
            attention_mask = tt[2].to(device)
            res = model(input_ids=token_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            rank[i % topx] = res.logits[0][0].cpu().item()
            if (i + 1) % topx == 0:
                ind, qName, _ = testset.get_query_doc_name(i)
                print(qName)
                rankStr = [str(r) for r in rank]
                resultList['query_id'].append(qName)
                resultList['doc_ids'].append(' '.join(data[ind]['bm25_top1000']))
                resultList['doc_scores'].append(' '.join(rankStr))
                rank = [0] * topx
    df = pd.DataFrame(resultList)
    df.to_csv(f'./ntust-ir2020-homework6/bert_result_{args.test}.csv', index=False)

def get_args():
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-mdl', '--model', default=None, dest='model', help='./models/epoch_0.pth')
    # parser.add_argument('-bs', '--batch_size', type=int, default=1, dest='batch_size')
    parser.add_argument('-test', '--test', type=str, required=True, dest='test')
    parser.add_argument('-fit_alpha', '--fit_alpha', action='store_true')
    parser.add_argument('-run_final', '--run_final', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    docs, test, train = get_csv_data()
    data = train if args.test == 'train' else test

    if args.model:
        predict_from_model(args, docs, data)
    alpha = 2
    if args.fit_alpha and args.test == 'train':
        alpha = test_alpha_by_use_map(args, docs, data)
    print(alpha)
    if args.run_final:
        get_final_result(args, docs, data, alpha=alpha)

    
