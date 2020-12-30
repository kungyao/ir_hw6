
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

    prompt = "TEST ABC"
    choice0 = "TEST_0"
    choice1 = "TEST_1"
    # choice2 = "TEST_2"
    # choice3 = "TEST_3"
    encoding = tokenizer(prompt, choice0, padding=True)
    encoding = tokenizer([[prompt, prompt], [choice0, choice1]], padding=True)
    print(encoding)
    import torch
    print(model.classifier)
    print(model(input_ids=torch.tensor([encoding['input_ids'], encoding['input_ids']])))
    # print(tokenizer(prompt, choice0, padding=True))
    # print(tokenizer(prompt, choice1, padding=True))
    # print(tokenizer(prompt, choice2, padding=True))
    # print(tokenizer(prompt, choice3, padding=True))