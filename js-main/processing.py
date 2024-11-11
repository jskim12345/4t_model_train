import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import build_dense_graph
# numpy와 pandas: 데이터 처리를 위해 사용됩니다.
# torch: PyTorch 라이브러리로, 딥러닝 모델 구축 및 학습에 사용됩니다.
# torch.utils.data: 데이터셋 및 데이터로더를 구축하는 데 사용되는 유틸리티를 제공합니다.
# torch.nn.utils.rnn: 순환 신경망(RNN)에서 패딩 작업에 사용됩니다.
# utils: 외부 모듈에서 build_dense_graph 함수를 가져옵니다.

# KTDataset: PyTorch의 Dataset을 상속받은 클래스입니다.
# init: 특징(features), 질문(questions), 정답(answers)을 초기화합니다.
# getitem: 인덱스를 기반으로 특정 데이터 항목을 반환합니다.
# len: 데이터셋의 크기를 반환합니다.
class KTDataset(Dataset):
    def __init__(self, features, questions, answers, user_ids):
        super(KTDataset, self).__init__()
        self.features = features
        self.questions = questions
        self.answers = answers
        self.user_ids = user_ids  # user_id 추가

    def __getitem__(self, index):
        return self.features[index], self.questions[index], self.answers[index], self.user_ids[index]  # user_id 반환

    def __len__(self):
        return len(self.features)

# pad_collate: 배치 데이터를 패딩하는 함수입니다.
# zip(*batch)를 사용하여 배치에서 특징, 질문 및 정답을 분리합니다.
# 각 리스트를 텐서로 변환한 후, pad_sequence를 사용하여 길이가 다른 시퀀스를 동일한 길이로 맞춥니다.
# 패딩된 텐서를 반환합니다.
def pad_collate(batch):
    (features, questions, answers, user_ids) = zip(*batch)  # user_ids를 포함
    features = [torch.LongTensor(feat) for feat in features]
    questions = [torch.LongTensor(qt) for qt in questions]
    answers = [torch.LongTensor(ans) for ans in answers]
    feature_pad = pad_sequence(features, batch_first=True, padding_value=-1)
    question_pad = pad_sequence(questions, batch_first=True, padding_value=-1)
    answer_pad = pad_sequence(answers, batch_first=True, padding_value=-1)
    return feature_pad, question_pad, answer_pad, user_ids  # user_ids도 반환

# load_dataset: 데이터셋을 로드하고 전처리하는 함수입니다.
# 인자로 데이터 파일 경로, 배치 크기, 그래프 유형, 그래프 경로, 학습/검증 비율 등을 받습니다.
def load_dataset(file_path, batch_size, graph_type, dkt_graph_path=None, train_ratio=0.7, val_ratio=0.2, shuffle=True, model_type='GKT', use_binary=True, res_len=2, use_cuda=True):
    print('load_s')
    r"""
    Parameters:
        file_path: input file path of knowledge tracing data
        batch_size: the size of a student batch
        graph_type: the type of the concept graph
        shuffle: whether to shuffle the dataset or not
        use_cuda: whether to use GPU to accelerate training speed
    Return:
        concept_num: the number of all concepts(or questions)
        graph: the static graph is graph type is in ['Dense', 'Transition', 'DKT'], otherwise graph is None
        train_data_loader: data loader of the training dataset
        valid_data_loader: data loader of the validation dataset
        test_data_loader: data loader of the test dataset
    NOTE: stole some code from https://github.com/lccasagrande/Deep-Knowledge-Tracing/blob/master/deepkt/data_util.py
    """
    df = file_path
    # if "skill_id" not in df.columns:
    #     raise KeyError(f"The column 'skill_id' was not found on {file_path}")
    # if "correct" not in df.columns:
    #     raise KeyError(f"The column 'correct' was not found on {file_path}")
    # if "user_id" not in df.columns:
    #     raise KeyError(f"The column 'user_id' was not found on {file_path}")

    # # if not (df['correct'].isin([0, 1])).all():
    # #     raise KeyError(f"The values of the column 'correct' must be 0 or 1.")

    # # Step 1.1 - Remove questions without skill
    # df.dropna(subset=['skill_id'], inplace=True)

    # # Step 1.2 - Remove users with a single answer
    # df = df.groupby('user_id').filter(lambda q: len(q) > 1).copy()

    # # Step 2 - Enumerate skill id
    # df['skill'], skill_mapping = pd.factorize(df['skill_id'], sort=True)  # we can also use problem_id to represent exercises

    # # Step 3 - Cross skill id with answer to form a synthetic feature
    # # use_binary: (0,1); !use_binary: (1,2,3,4,5,6,7,8,9,10,11,12). Either way, the correct result index is guaranteed to be 1
    # if use_binary:
    #     df['skill_with_answer'] = df['skill'] * 2 + df['correct']
    # else:
    #     df['skill_with_answer'] = df['skill'] * res_len + df['correct'] - 1


    # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    feature_list = []
    question_list = []
    answer_list = []
    seq_len_list = []
    user_id_list = []

    def get_data(series):
        feature_list.append(series['skill_with_answer'].tolist())
        question_list.append(series['skill'].tolist())
        answer_list.append(series['correct'].eq(1).astype('int').tolist())
        seq_len_list.append(series['correct'].shape[0])
        user_id_list.append(series.name)  # user_id 저장

    df.groupby('user_id').apply(get_data)
    max_seq_len = np.max(seq_len_list)
    print('max seq_len: ', max_seq_len)
    student_num = len(seq_len_list)
    print('student num: ', student_num)
    feature_dim = int(df['skill_with_answer'].max() + 1)
    print('feature_dim: ', feature_dim)
    question_dim = int(df['skill'].max() + 1)
    print('question_dim: ', question_dim)
    concept_num = question_dim

    # df.to_csv('df.csv', index=False) 
    
    # print('feature_dim:', feature_dim, 'res_len*question_dim:', res_len*question_dim)
    # assert feature_dim == res_len * question_dim

    kt_dataset = KTDataset(feature_list, question_list, answer_list, user_id_list)
    train_size = int(train_ratio * student_num)
    val_size = int(val_ratio * student_num)
    test_size = student_num - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(kt_dataset, [train_size, val_size, test_size])
    print('train_size: ', train_size, 'val_size: ', val_size, 'test_size: ', test_size)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    valid_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)

    graph = None
    if model_type == 'GKT':
        if graph_type == 'Dense':
            graph = build_dense_graph(concept_num)
        elif graph_type == 'Transition':
            graph = build_transition_graph(question_list, seq_len_list, train_dataset.indices, student_num, concept_num)
        elif graph_type == 'DKT':
            graph = build_dkt_graph(dkt_graph_path, concept_num)
        if use_cuda and graph_type in ['Dense', 'Transition', 'DKT']:
            graph = graph.cuda()
    print('load_ss')
    return concept_num, graph, train_data_loader, valid_data_loader, test_data_loader

# build_transition_graph: 질문 간의 전이 그래프를 구축하는 함수입니다.
# 각 학생의 질문 시퀀스를 기반으로 그래프를 구성합니다.
def build_transition_graph(question_list, seq_len_list, indices, student_num, concept_num):
    graph = np.zeros((concept_num, concept_num))
    student_dict = dict(zip(indices, np.arange(student_num)))
    for i in range(student_num):
        if i not in student_dict:
            continue
        questions = question_list[i]
        seq_len = seq_len_list[i]
        for j in range(seq_len - 1):
            pre = questions[j]
            next = questions[j + 1]
            graph[pre, next] += 1
    np.fill_diagonal(graph, 0)
    # row normalization
    rowsum = np.array(graph.sum(1))
    def inv(x):
        if x == 0:
            return x
        return 1. / x
    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum).flatten()
    r_mat_inv = np.diag(r_inv)
    graph = r_mat_inv.dot(graph)
    # covert to tensor
    graph = torch.from_numpy(graph).float()
    return graph

# build_dkt_graph: DKT(Dynamic Key-Value Memory Networks) 그래프를 구축하는 함수입니다.
def build_dkt_graph(file_path, concept_num):
    graph = np.loadtxt(file_path)
    assert graph.shape[0] == concept_num and graph.shape[1] == concept_num
    graph = torch.from_numpy(graph).float()
    return graph