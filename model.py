
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
    print(tokenizer.bos_token_id)
    print(tokenizer.sep_token_id)
