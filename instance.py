import torch
import pickle
import torch.utils.data as Data
from preprocess.data_loader_feature import get_feature
from configuration import config as cf
from model.bilinear import MAPEP
import torch.nn.functional as F
import pandas as pd

import random
import numpy as np

# 固定随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 加载 token 到 index 的映射表
token2index = pickle.load(open('./data/residue2idx.pkl', 'rb'))


def transform_token2index(sequences):
    for i, seq in enumerate(sequences):
        sequences[i] = list(seq)

    token_list = list()
    max_len = 0
    for seq in sequences:
        seq_id = [token2index[residue] for residue in seq]
        token_list.append(seq_id)
        if len(seq) > max_len:
            max_len = len(seq)

    return token_list, max_len


def make_data_with_unified_length(token_list, max_len):
    max_len = 52 + 2  # add [CLS] and [SEP]
    data = []
    for i in range(len(token_list)):
        token_list[i] = [token2index['[CLS]']] + token_list[i] + [token2index['[SEP]']]
        n_pad = max_len - len(token_list[i])
        token_list[i].extend([0] * n_pad)
        data.append(token_list[i])
    return data


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]


def construct_dataset(data):
    batch_size = 32
    input_ids = data
    input_ids = torch.LongTensor(input_ids)
    data_loader = Data.DataLoader(MyDataSet(input_ids), batch_size=batch_size, drop_last=False, shuffle=False)
    return data_loader


def construct_dataset_fc(data):
    batch_size = 32
    feature = data
    feature = torch.FloatTensor(feature)
    data_loader = Data.DataLoader(MyDataSet(feature), batch_size=batch_size, drop_last=False, shuffle=False)
    return data_loader


def load_config():
    config = cf.get_train_config()
    config.max_len = 54
    return config


def process(sequences):
    sequences_o = sequences.copy()
    token_list, max_len = transform_token2index(sequences)
    data_sample = make_data_with_unified_length(token_list, max_len)
    data_loader_sample = construct_dataset(data_sample)
    data_sample_fc = get_feature(sequences_o, {'morgan', 'aaindex'})
    data_loader_sample_fc = construct_dataset_fc(data_sample_fc)

    return data_loader_sample, data_loader_sample_fc


def predict_sequences(model, data_loader, data_loader_fc, device):
    """
    批量预测多个序列，并返回每个序列的预测结果。
    """
    results = []
    model.eval()
    with torch.no_grad():
        for input, input_fc in zip(data_loader, data_loader_fc):
            input = input.to(device)
            input_fc = input_fc.to(device)
            output, _ = model(input, input_fc)
            pred_prob_all = F.softmax(output, dim=1)
            # 提取正类概率，逐个添加到结果列表
            pred_prob_positive = pred_prob_all[:, 1].cpu().numpy()
            results.extend(pred_prob_positive)  # 展平结果
    return results


if __name__ == '__main__':
    # 直接从当前工作目录读取固定文件名
    input_file = './input.txt'  # 假设文件名为 input.txt
    with open(input_file, 'r') as f:
        peptide_sequences = [line.strip() for line in f.readlines()]
    peptide_sequences_copy = peptide_sequences.copy()

    # 配置模型
    config = load_config()
    config.vocab_size = len(token2index)

    # 加载模型
    model = MAPEP(config)
    state_dict = torch.load('/home/liangxiao/MA-PEP/ME-PEP_model.pt')
    model.load_state_dict(state_dict)
    device = torch.device('cuda')
    model = model.to(device)

    # 数据预处理
    data_loader_sample, data_loader_sample_fc = process(peptide_sequences)

    # 批量预测
    predictions = predict_sequences(model, data_loader_sample, data_loader_sample_fc, device)

    # 打印结果并保存到文件
    results_df = pd.DataFrame({
        'sequence': peptide_sequences_copy,
        'prediction_probability': predictions
    })

    print('*' * 10 + ' Prediction Results ' + '*' * 10)
    print(results_df)

    # 保存到文件
    results_df.to_csv('output.csv', index=False)
    print("Results saved to predictions.csv")
