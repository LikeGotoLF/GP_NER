import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from model import GlobalPointer
from ner_base_pt import load_data, NamedEntityRecognizer
from transformers import BertTokenizerFast,BertModel
import torch


model_name = 'hfl/chinese-roberta-wwm-ext'
tokenizer = BertTokenizerFast.from_pretrained(model_name)
# model = GlobalPointer(BertModel.from_pretrained(model_name),len(categories),64)
device = torch.device('cuda')
model = GlobalPointer(BertModel.from_pretrained(model_name),10,64).to(device)
model.load_state_dict(torch.load('model_states/clue20.pt'))
torch.save(model.encoder.state_dict(), 'model_states/bert/clue_bert.pt')