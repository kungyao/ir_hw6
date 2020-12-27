
def get_albert_model_and_tokenizer():
    from transformers import AlbertTokenizer, AlbertForSequenceClassification
    model = AlbertForSequenceClassification.from_pretrained('albert-base-v2')
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    return model, tokenizer

def get_roberta_model_and_tokenizer():
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    return model, tokenizer

if __name__ == '__main__':
    import torch
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    print(f'Use {device} to training')


    model, tokenizer = get_albert_model_and_tokenizer()
    print(tokenizer.pad_token_id)
    print(tokenizer.bos_token_id)
    print(tokenizer.sep_token_id)

    query = 'Poliomyelitis and Post-Polio'
    content = 'fuck.'
    content = 'of statement by Bishop Samuel Ruiz </F> Garcia to unidentified domestic and foreign reporters at the San Cristobal de las Casas Cathedral on 27 February] [Text] Good evening to all the domestic and international sisters and brothers of the press, radio, and television media. Events that occur outside the discussion table influence differently those who are sitting at this table and on those who are abroad. Although we cannot say news has been lacking, it has certainly influenced the mood of the talks. For example, the news on what has been happening in Altamirano has influenced our work. Even if those events would not have occurred and we would not have known of the steps taken to achieve greater peace, it would have been difficult for the parties at the table to continue to progress without feeling there was a disturbing external pressure.of statement by Bishop Samuel Ruiz </F> Garcia to unidentified domestic and foreign reporters at the San Cristobal de las Casas Cathedral on 27 February] [Text] Good evening to all the domestic and international sisters and brothers of the press, radio, and television media. Events that occur outside the discussion table influence differently those who are sitting at this table and on those who are abroad. Although we cannot say news has been lacking, it has certainly influenced the mood of the talks. For example, the news on what has been happening in Altamirano has influenced our work. Even if those events would not have occurred and we would not have known of the steps taken to achieve greater peace, it would have been difficult for the parties at the table to continue to progress without feeling there was a disturbing external pressure.of statement by Bishop Samuel Ruiz </F> Garcia to unidentified domestic and foreign reporters at the San Cristobal de las Casas Cathedral on 27 February] [Text] Good evening to all the domestic and international sisters and brothers of the press, radio, and television media. Events that occur outside the discussion table influence differently those who are sitting at this table and on those who are abroad. Although we cannot say news has been lacking, it has certainly influenced the mood of the talks. For example, the news on what has been happening in Altamirano has influenced our work. Even if those events would not have occurred and we would not have known of the steps taken to achieve greater peace, it would have been difficult for the parties at the table to continue to progress without feeling there was a disturbing external pressure.of statement by Bishop Samuel Ruiz </F> Garcia to unidentified domestic and foreign reporters at the San Cristobal de las Casas Cathedral on 27 February] [Text] Good evening to all the domestic and international sisters and brothers of the press, radio, and television media. Events that occur outside the discussion table influence differently those who are sitting at this table and on those who are abroad. Although we cannot say news has been lacking, it has certainly influenced the mood of the talks. For example, the news on what has been happening in Altamirano has influenced our work. Even if those events would not have occurred and we would not have known of the steps taken to achieve greater peace, it would have been difficult for the parties at the table to continue to progress without feeling there was a disturbing external pressure.of statement by Bishop Samuel Ruiz </F> Garcia to unidentified domestic and foreign reporters at the San Cristobal de las Casas Cathedral on 27 February] [Text] Good evening to all the domestic and international sisters and brothers of the press, radio, and television media. Events that occur outside the discussion table influence differently those who are sitting at this table and on those who are abroad. Although we cannot say news has been lacking, it has certainly influenced the mood of the talks. For example, the news on what has been happening in Altamirano has influenced our work. Even if those events would not have occurred and we would not have known of the steps taken to achieve greater peace, it would have been difficult for the parties at the table to continue to progress without feeling there was a disturbing external pressure.of statement by Bishop Samuel Ruiz </F> Garcia to unidentified domestic and foreign reporters at the San Cristobal de las Casas Cathedral on 27 February] [Text] Good evening to all the domestic and international sisters and brothers of the press, radio, and television media. Events that occur outside the discussion table influence differently those who are sitting at this table and on those who are abroad. Although we cannot say news has been lacking, it has certainly influenced the mood of the talks. For example, the news on what has been happening in Altamirano has influenced our work. Even if those events would not have occurred and we would not have known of the steps taken to achieve greater peace, it would have been difficult for the parties at the table to continue to progress without feeling there was a disturbing external pressure.of statement by Bishop Samuel Ruiz </F> Garcia to unidentified domestic and foreign reporters at the San Cristobal de las Casas Cathedral on 27 February] [Text] Good evening to all the domestic and international sisters and brothers of the press, radio, and television media. Events that occur outside the discussion table influence differently those who are sitting at this table and on those who are abroad. Although we cannot say news has been lacking, it has certainly influenced the mood of the talks. For example, the news on what has been happening in Altamirano has influenced our work. Even if those events would not have occurred and we would not have known of the steps taken to achieve greater peace, it would have been difficult for the parties at the table to continue to progress without feeling there was a disturbing external pressure.of statement by Bishop Samuel Ruiz </F> Garcia to unidentified domestic and foreign reporters at the San Cristobal de las Casas Cathedral on 27 February] [Text] Good evening to all the domestic and international sisters and brothers of the press, radio, and television media. Events that occur outside the discussion table influence differently those who are sitting at this table and on those who are abroad. Although we cannot say news has been lacking, it has certainly influenced the mood of the talks. For example, the news on what has been happening in Altamirano has influenced our work. Even if those events would not have occurred and we would not have known of the steps taken to achieve greater peace, it would have been difficult for the parties at the table to continue to progress without feeling there was a disturbing external pressure.of statement by Bishop Samuel Ruiz </F> Garcia to unidentified domestic and foreign reporters at the San Cristobal de las Casas Cathedral on 27 February] [Text] Good evening to all the domestic and international sisters and brothers of the press, radio, and television media. Events that occur outside the discussion table influence differently those who are sitting at this table and on those who are abroad. Although we cannot say news has been lacking, it has certainly influenced the mood of the talks. For example, the news on what has been happening in Altamirano has influenced our work. Even if those events would not have occurred and we would not have known of the steps taken to achieve greater peace, it would have been difficult for the parties at the table to continue to progress without feeling there was a disturbing external pressure.of statement by Bishop Samuel Ruiz </F> Garcia to unidentified domestic and foreign reporters at the San Cristobal de las Casas Cathedral on 27 February] [Text] Good evening to all the domestic and international sisters and brothers of the press, radio, and television media. Events that occur outside the discussion table influence differently those who are sitting at this table and on those who are abroad. Although we cannot say news has been lacking, it has certainly influenced the mood of the talks. For example, the news on what has been happening in Altamirano has influenced our work. Even if those events would not have occurred and we would not have known of the steps taken to achieve greater peace, it would have been difficult for the parties at the table to continue to progress without feeling there was a disturbing external pressure.'

    word_pieces = ['[CLS]']
    qtoken = tokenizer.tokenize(query)
    word_pieces += qtoken + ["[SEP]"]
    len_a = len(word_pieces)

    dtoken = tokenizer.tokenize(content)
    word_pieces += dtoken
    if len(word_pieces) >= 512:
        word_pieces = word_pieces[:511]
    word_pieces += ["[SEP]"]

    print(len(word_pieces))

    len_b = len(word_pieces) - len_a

    token_ids = tokenizer.convert_tokens_to_ids(word_pieces)
    token_type_ids = [0] * len_a + [1] * len_b
    # tokens = tokenizer(f'{query}[SEP]{content}', max_length=512, truncation=True, add_special_tokens=True)
    print(token_ids, token_type_ids)

    # # split data and to tensor
    # token_ids = torch.tensor([tokens['input_ids']])
    # atten_masks = torch.tensor([tokens['attention_mask']])
    # labels = torch.tensor([1])

    # # print(token_ids)
    # print(token_ids.shape)
    # print(atten_masks.shape)

    # model.to(device)
    # token_ids = token_ids.to(device)
    # atten_masks = atten_masks.to(device)
    # labels = labels.to(device)

    # res = model(input_ids=token_ids, attention_mask=atten_masks, labels=labels)
    # print(res)