from transformers import RobertaTokenizer, RobertaForSequenceClassification

def get_roberta_model_and_tokenizer():
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    return model, tokenizer
