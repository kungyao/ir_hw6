import os
import argparse

import torch
from torch.utils.data import DataLoader

from data import CorpusSet, collate_fn, get_csv_data
from model import get_roberta_model_and_tokenizer

def get_args():
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-ep', '--epoch', type=int, default=15, dest='epoch')
    parser.add_argument('-bs', '--batch_size', type=int, default=1, dest='batch_size')
    parser.add_argument('-o', '--output', type=str, default='./models', dest='output')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    print(f'Use {device} to training')

    # load data
    docs, test, train, neg_train = get_csv_data()

    # get model
    model, tokenizer = get_roberta_model_and_tokenizer()

    # construct data loader
    trainset = CorpusSet(tokenizer, docs, train, neg_doc=neg_train, mode='train')
    trainloader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4, shuffle=True)

    # testset = CorpusSet(tokenizer, docs, train, neg_doc=neg_train, mode='test')
    # testloader = DataLoader(testset, batch_size=1, collate_fn=collate_fn, num_workers=4)

    model.to(device)
    for _ in range(args.epoch):
        for i, data in enumerate(trainloader):
            token_ids = data[0].to(device)
            atten_masks = data[1].to(device)
            labels = data[2].to(device)

            res = model(input_ids=token_ids, attention_mask=atten_masks, labels=labels)

            loss = res.loss
            logits = res.logits
        torch.save(model, os.path.join(args.output, f'epoch_{epoch}.mdl'))

