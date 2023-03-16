import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import ljqpy
import torch
from transformers import BertTokenizer, BertModel, BertTokenizerFast
from torch.utils.data import DataLoader, Dataset
from model import GlobalPointer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from tqdm import tqdm
import pt_utils


categories = set()

def load_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
    """
    res = []
    data = ljqpy.LoadJsons(filename)
    for l in data:
        d = [l['text']]
        for k, v in l['label'].items():
            categories.add(k)
            for spans in v.values():
                for start, end in spans:
                    d.append((start, end, k))
        res.append(d)
    return res

data_dir = '/mnt/data122/qsm/NER/cluener'
train_data = load_data(os.path.join(data_dir,'train.json'))
val_data = load_data(os.path.join(data_dir,'dev.json'))

id2label = list(categories)
label2id = {l:i for i,l in enumerate(id2label)}
model_name = 'hfl/chinese-roberta-wwm-ext'
tokenizer = BertTokenizerFast.from_pretrained(model_name)
# model = GlobalPointer(BertModel.from_pretrained(model_name),len(categories),64)
device = torch.device('cuda')

class MyDataset(Dataset):
    def __init__(self,data):
        super().__init__()
        self.data = data
        # self.label2id = label2id

    def __getitem__(self,index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
def collate_fn(batch):
    inputs = tokenizer([d[0] for d in batch],return_tensors='pt',truncation=True, max_length=256,\
                  padding=True,return_offsets_mapping=True)
    xx = inputs.input_ids
    mapping = inputs.offset_mapping
    label_list = [torch.zeros(len(categories),xx.shape[1],xx.shape[1]) for _ in range(xx.shape[0])]
    for i,d in enumerate(batch):
        start_mapping = {j[0].item(): k for k, j in enumerate(mapping[i])} # char_index_start to token_index
        end_mapping = {j[-1].item(): k for k, j in enumerate(mapping[i])}
        # print(start_mapping,end_mapping)
        for start, end, label in d[1:]:
            # print(start,end,label)
            if start in start_mapping and end+1 in end_mapping:
                start = start_mapping[start]
                end = end_mapping[end+1]
                id = label2id[label]
                label_list[i][id,start, end] = 1
    
    return (inputs.input_ids,inputs.attention_mask,inputs.token_type_ids,torch.stack(label_list,0))

train_ds = MyDataset(train_data)
train_dl = DataLoader(train_ds,shuffle=True,collate_fn=collate_fn,batch_size=32)


def multilabel_categorical_crossentropy(y_true, y_pred):
    """
    https://kexue.fm/archives/7359
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()


def loss_fun(y_true, y_pred):
        """
        y_true:(batch_size, ent_type_size, seq_len, seq_len)
        y_pred:(batch_size, ent_type_size, seq_len, seq_len)
        """
        batch_size, ent_type_size = y_pred.shape[:2]
        y_true = y_true.reshape(batch_size * ent_type_size, -1)
        y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
        loss = multilabel_categorical_crossentropy(y_true, y_pred)
        return loss

record = {'train_loss':[],'val_score':[]}

def train_func(model, ditem):
    xx, att_mask,tti, yy = ditem[0].to(device),ditem[1].to(device),ditem[2].to(device),ditem[-1].to(device)
    zz = model(xx,att_mask,tti)
    loss = loss_fun(yy,zz)
    # print(yy)
    # print(zz)
    record["train_loss"].append(loss.item())
    return {'loss': loss}


def test_func(): 
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(val_data[:50], ncols=100):
        # print(d[0])
        R = set(NER.recognize(d[0]))
        T = set([tuple(i) for i in d[1:]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    print(f'Prec: {precision:.4f},  Reca: {recall:.4f},  F1: {f1:.4f}')
    record['val_score'].append(f1)
    model.train()
    return precision, recall,f1

class NamedEntityRecognizer(object):
    """命名实体识别器
    对于GP模型返回 (n, maxlen, maxlen) 
    返回: 同 data 中定义的 (start, end, label) 形式
    """
    def recognize(self, text, threshold=0):
        inputs = tokenizer(text,return_tensors='pt',truncation=True, max_length=256,return_offsets_mapping=True).to(device)
        mapping = inputs.offset_mapping[0]
        scores = model(inputs.input_ids,inputs.attention_mask,inputs.token_type_ids)[0].cpu()
        # print(scores.shape)
        l = len(inputs.input_ids)
        # mask = torch.triu(torch.ones(l,l)).unsqueeze(0).expand(scores.shape)
        # scores = scores*mask
        scores[:, [0, -1]] -= np.inf # start 不能为 [CLS], [SEP]
        scores[:, :, [0, -1]] -= np.inf # end 不能为 [CLS], [SEP]
        # scores = scores.cpu()
        entities = []
        # print(scores.shape)
        for l, start, end in zip(*torch.where(scores > threshold)):
            entities.append((mapping[start][0].item(), mapping[end][-1].item()-1, id2label[l]))
        # print(entities)
        return entities

NER = NamedEntityRecognizer()

def plot_learning_curve(record,pic_n):
    y1 = record['train_loss']
    y2 = record['val_score']
    x1 = np.arange(1,len(y1)+1)
    x2 = x1[::int(len(y1)/len(y2))]
    fig = figure(figsize = (6,4))
    ax1 = fig.add_subplot(111)
    ax1.plot(x1,y1, c = 'tab:red', label = 'train_loss')
    ax2 = ax1.twinx()
    ax2.plot(x2,y2, c='tab:cyan', label='val_score')
    ax1.set_xlabel('steps')
    ax1.set_ylabel('train_loss')
    ax2.set_ylabel('val_score')
    plt.title('Learning curve')
    ax1.legend(loc=1)
    ax2.legend(loc=2)
    # plt.show()
    plt.savefig(pic_n)
    return

def train_model(model, optimizer, train_dl, epochs=3, train_func=None, test_func=None, 
                scheduler=None, save_file=None, accelerator=None, epoch_len=None):  # accelerator：适用于多卡的机器，epoch_len到该epoch提前停止
    best_f1 = -1
    for epoch in range(epochs):
        model.train()
        print(f'\nEpoch {epoch+1} / {epochs}:')
        if accelerator:
            pbar = tqdm(train_dl, total=epoch_len, disable=not accelerator.is_local_main_process)
        else: 
            pbar = tqdm(train_dl, total=epoch_len)
        metricsums = {}
        iters, accloss = 0, 0
        for ditem in pbar:
            metrics = {}
            loss = train_func(model, ditem)
            if type(loss) is type({}):
                metrics = {k:v.detach().mean().item() for k,v in loss.items() if k != 'loss'}
                loss = loss['loss']
            iters += 1; accloss += loss
            optimizer.zero_grad()
            if accelerator: 
                accelerator.backward(loss)
            else: 
                loss.backward()
            optimizer.step()
            if scheduler:
                if accelerator is None or not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
            for k, v in metrics.items(): metricsums[k] = metricsums.get(k,0) + v
            infos = {'loss': f'{accloss/iters:.4f}'}
            for k, v in metricsums.items(): infos[k] = f'{v/iters:.4f}' 
            pbar.set_postfix(infos)
            if epoch_len and iters > epoch_len: break
        pbar.close()
        if test_func:
            if accelerator is None or accelerator.is_local_main_process: 
                model.eval()
                prec,reca,f1 = test_func()
                if f1 >=best_f1 and save_file:
                    if accelerator:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        accelerator.save(unwrapped_model.state_dict(), save_file)
                    else:
                        torch.save(model.state_dict(), save_file)
                    print(f"Epoch {epoch + 1}, best model saved. (Prec: {prec:.4f},  Reca: {reca:.4f},  F1: {f1:.4f})")
                    best_f1 = f1

if __name__ == '__main__':
    mfile = 'clue20.pt'
    epochs = 20
    total_steps = len(train_dl) * epochs
    model = GlobalPointer(BertModel.from_pretrained(model_name),len(categories),64).to(device)
    optimizer, scheduler = pt_utils.get_bert_optim_and_sche(model, 3e-5, total_steps)
    # train_model()
    train_model(model, optimizer, train_dl, epochs, train_func, test_func, scheduler=scheduler, save_file=mfile)
    plot_learning_curve(record,'pic_clue20')












