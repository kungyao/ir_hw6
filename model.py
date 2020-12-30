
def get_bert_model_and_tokenizer(ifModel=True):
    from transformers import BertTokenizer, BertForMultipleChoice
    if ifModel:
        model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
    else:
        model = None
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

if __name__ == '__main__':
    import torch
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    print(f'Use {device} to training')

    model, tokenizer = get_bert_model_and_tokenizer()
    print(tokenizer.pad_token_id)
    print(tokenizer.cls_token_id)
    print(tokenizer.sep_token_id)

    prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    choice0 = "It is eaten with a fork and a knife."
    choice1 = "It is eaten while held in the hand."
    labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

    encoding = tokenizer([[prompt, prompt], [choice0, choice1]], padding=True)

    testSingle = {
        'input_ids'         : torch.tensor([[encoding['input_ids'][0]], [encoding['input_ids'][1]]]).to(device),
        'token_type_ids'    : torch.tensor([[encoding['token_type_ids'][0]], [encoding['token_type_ids'][1]]]).to(device),
        'attention_mask'    : torch.tensor([[encoding['attention_mask'][0]], [encoding['attention_mask'][1]]]).to(device),
        'labels'            : torch.tensor([0, 0]).to(device),
    }

    print(testSingle['input_ids'].shape)
    print(testSingle['token_type_ids'].shape)
    print(testSingle['attention_mask'].shape)
    print(testSingle['labels'].shape)

    encoding['input_ids'] = torch.tensor([encoding['input_ids']])
    encoding['token_type_ids'] = torch.tensor([encoding['token_type_ids']])
    encoding['attention_mask'] = torch.tensor([encoding['attention_mask']])

    with torch.no_grad():
        # batch size
        # same as third type of print
        # print(model(**{k: v for k,v in encoding.items()}, labels=labels))
        print(model(input_ids=encoding['input_ids'], labels=labels))
        print(model(input_ids=encoding['input_ids'], token_type_ids=encoding['token_type_ids'], labels=labels))
        print(model(input_ids=encoding['input_ids'], token_type_ids=encoding['token_type_ids'], attention_mask=encoding['attention_mask'], labels=labels))

    model.to(device)
    encoding['input_ids'] = encoding['input_ids'].to(device)
    encoding['token_type_ids'] = encoding['token_type_ids'].to(device)
    encoding['attention_mask'] = encoding['attention_mask'].to(device)
    labels = labels.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=3e-5)
    for i in range(10):
        res = model(input_ids=encoding['input_ids'], token_type_ids=encoding['token_type_ids'], attention_mask=encoding['attention_mask'], labels=labels)

        optim.zero_grad()
        res.loss.backward()
        optim.step()

        print(f'epoch {i}, loss {res.loss}')

    print('multi test')
    with torch.no_grad():
        # same as third type of print
        # print(model(**{k: v for k,v in encoding.items()}, labels=labels))
        print(model(input_ids=encoding['input_ids'], labels=labels))
        print(model(input_ids=encoding['input_ids'], token_type_ids=encoding['token_type_ids'], labels=labels))
        print(model(input_ids=encoding['input_ids'], token_type_ids=encoding['token_type_ids'], attention_mask=encoding['attention_mask'], labels=labels))
    
    # to single data
    print('single test')
    with torch.no_grad():
        print(model(input_ids=testSingle['input_ids'], token_type_ids=testSingle['token_type_ids'], attention_mask=testSingle['attention_mask']))
    