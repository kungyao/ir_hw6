import os
import argparse

import torch
from torch.utils.data import DataLoader

from data import CorpusSet, collate_fn, get_csv_data, preprocess_df
from model import get_bert_model_and_tokenizer

def get_args():
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-ep', '--epoch', type=int, default=5, dest='epoch')
    parser.add_argument('-bs', '--batch_size', type=int, default=2, dest='batch_size')
    parser.add_argument('-o', '--output', type=str, default='./models', dest='output')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    print(f'Use {device} to training')

    # load data
    docs, _, train = get_csv_data()

    # get model
    model, tokenizer = get_bert_model_and_tokenizer()

    optim = torch.optim.Adam(model.parameters(), lr=3e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.1)

    # construct data loader
    _, train = preprocess_df(docs, None, train, tokenizer)
    trainset = CorpusSet(train, mode='train')
    trainloader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

    print_thres = 100
    model.to(device)
    for ep in range(args.epoch):
        print(f'training epoch {ep}')
        total_loss = 0
        for i, data in enumerate(trainloader):
            token_ids = data[0].to(device)
            token_type_ids = data[1].to(device)
            atten_masks = data[2].to(device)
            labels = data[3].to(device)

            res = model(input_ids=token_ids, token_type_ids=token_type_ids, attention_mask=atten_masks, labels=labels)

            optim.zero_grad()
            res.loss.backward()
            optim.step()

            total_loss += res.loss
            # loss = res.loss
            # logits = res.logits
            if (i + 1) % print_thres == 0:
                print(f'epoch {ep}, batch {i + 1}, loss {total_loss / print_thres}')
                total_loss = 0
        torch.save(model, os.path.join(args.output, f'epoch_{ep}.mdl'))

