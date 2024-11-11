import numpy as np
import time
import random
import argparse
import pickle
import os
import gc
import datetime
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from models import GKT, MultiHeadAttention, VAE, DKT
from metrics import KTLoss, VAELoss
from processing import load_dataset
import boto3
import mlflow
from io import BytesIO
from dotenv import load_dotenv
import pandas as pd
from io import StringIO
import io


# Load environment variables
load_dotenv()
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
mlflow.set_tracking_uri(mlflow_tracking_uri)
# AWS S3 설정

BUCKET_NAME = "result-4t"                  
MODEL_PREFIX = "mlflow/gkt_mlflow/"              # S3에 저장된 파일 경로
RESULT_PREFIX = "mlflow/gkt_mlflow/"
RAW_PREFIX = "mlflow/gkt_mlflow/df.csv"


s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)


mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', "http://43.203.12.181:5000")

experiment_name = "new_experiment_name"
experiment_id = "745537454314509167"
mlflow.set_tracking_uri(mlflow_tracking_uri)

mlflow.set_experiment(experiment_name)  

mlflow.start_run()

# 하이퍼파라미터 설정
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_false', default=True, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--data-dir', type=str, default='data', help='Data dir for loading input data.')
parser.add_argument('--data-file', type=str, default=RAW_PREFIX, help='Name of input data file.')
parser.add_argument('--save-dir', type=str, default='logs', help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('-graph-save-dir', type=str, default='graphs', help='Dir for saving concept graphs.')
parser.add_argument('--load-dir', type=str, default='', help='Where to load the trained model if finetunning. ' + 'Leave empty to train from scratch')
parser.add_argument('--dkt-graph-dir', type=str, default='dkt-graph', help='Where to load the pretrained dkt graph.')
parser.add_argument('--dkt-graph', type=str, default='dkt_graph.txt', help='DKT graph data file name.')
parser.add_argument('--model', type=str, default='GKT', help='Model type to use, support GKT and DKT.')
parser.add_argument('--hid-dim', type=int, default=32, help='Dimension of hidden knowledge states.')
parser.add_argument('--emb-dim', type=int, default=32, help='Dimension of concept embedding.')
parser.add_argument('--attn-dim', type=int, default=32, help='Dimension of multi-head attention layers.')
parser.add_argument('--vae-encoder-dim', type=int, default=32, help='Dimension of hidden layers in vae encoder.')
parser.add_argument('--vae-decoder-dim', type=int, default=32, help='Dimension of hidden layers in vae decoder.')
parser.add_argument('--edge-types', type=int, default=2, help='The number of edge types to infer.')
parser.add_argument('--graph-type', type=str, default='Dense', help='The type of latent concept graph.')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--bias', type=bool, default=True, help='Whether to add bias for neural network layers.')
parser.add_argument('--binary', type=bool, default=True, help='Whether only use 0/1 for results.')
parser.add_argument('--result-type', type=int, default=12, help='Number of results types when multiple results are used.')
parser.add_argument('--temp', type=float, default=0.5, help='Temperature for Gumbel softmax.')
parser.add_argument('--hard', action='store_true', default=False, help='Uses discrete samples in training forward pass.')
parser.add_argument('--no-factor', action='store_true', default=False, help='Disables factor graph model.')
parser.add_argument('--prior', action='store_true', default=False, help='Whether to use sparsity prior.')
parser.add_argument('--var', type=float, default=1, help='Output variance.')
# parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128, help='Number of samples per batch.')
# parser.add_argument('--batch-size', type=int, default=10, help='Number of samples per batch.')
parser.add_argument('--train-ratio', type=float, default=0.6, help='The ratio of training samples in a dataset.')
parser.add_argument('--val-ratio', type=float, default=0.2, help='The ratio of validation samples in a dataset.')
parser.add_argument('--shuffle', type=bool, default=True, help='Whether to shuffle the dataset or not.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--lr-decay', type=int, default=200, help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor.')
parser.add_argument('--test', type=bool, default=False, help='Whether to test for existed model.')
parser.add_argument('--test-model-dir', type=str, default='logs/expDKT', help='Existed model file dir.')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

res_len = 2 if args.binary else args.result_type

# Log hyperparameters to MLflow
mlflow.log_params(vars(args))

def load_s3_csv(bucket_name, prefix):
    response = s3_client.get_object(Bucket=bucket_name, Key=prefix)
    return pd.read_csv(BytesIO(response['Body'].read()))
# S3에서 데이터셋 로드
dataset_path = RAW_PREFIX
# S3에서 데이터셋 로드
df = load_s3_csv(BUCKET_NAME, RAW_PREFIX)
# 수정된 데이터 로드
concept_num, graph, train_loader, valid_loader, test_loader = load_dataset(
    df, args.batch_size, args.graph_type, train_ratio=args.train_ratio,
    val_ratio=args.val_ratio, shuffle=True, model_type=args.model, use_cuda=args.cuda
)

def save_pytorch_model_to_s3(model, s3_bucket, s3_key):
    """
    PyTorch 모델을 .pt 형식으로 S3에 저장하는 함수입니다.
    Parameters:
        model (torch.nn.Module): PyTorch 모델 객체
        s3_bucket (str): S3 버킷 이름
        s3_key (str): S3에 저장할 경로 (예: 'models/my_model.pt')
    """
    # S3 클라이언트 생성
    s3_client = boto3.client('s3')
    try:
        # 모델을 메모리에 저장하고 S3로 업로드
        with BytesIO() as model_buffer:
            torch.save(model.state_dict(), model_buffer)  # 모델의 가중치 저장
            model_buffer.seek(0)  # 시작 위치로 이동
            s3_client.put_object(Bucket=s3_bucket, Key=s3_key, Body=model_buffer.getvalue())
    except Exception as e:
        raise e

log = None
# save_dir = args.save_dir
save_dir = MODEL_PREFIX
if args.save_dir:
    exp_counter = 0
    now = datetime.datetime.now()
    # timestamp = now.isoformat()
    timestamp = now.strftime('%Y-%m-%d %H-%M-%S')
    if args.model == 'DKT':
        model_file_name = 'DKT'
    elif args.model == 'GKT':
        model_file_name = 'GKT' + '-' + args.graph_type
    else:
        raise NotImplementedError(args.model + ' model is not implemented!')
    save_dir = '{}/exp{}/'.format(args.save_dir, model_file_name + timestamp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    meta_file = os.path.join(save_dir, 'metadata.pkl')
    model_file = os.path.join(save_dir, model_file_name + '.pt')
    optimizer_file = os.path.join(save_dir, model_file_name + '-Optimizer.pt')
    scheduler_file = os.path.join(save_dir, model_file_name + '-Scheduler.pt')
    log_file = os.path.join(save_dir, 'log.txt')
    log = open(log_file, 'w')
    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_dir provided!" + "Testing (within this script) will throw an error.")


# dkt_graph_path를 확인하여 경로가 유효하지 않으면 None으로 설정합니다.
dkt_graph_path = os.path.join(args.dkt_graph_dir, args.dkt_graph)
if not os.path.exists(dkt_graph_path):
    dkt_graph_path = None

file_path = os.path.join(args.data_dir, args.data_file)
print(f"Attempting to load dataset from: {file_path}")

# build models
graph_model = None
if args.model == 'GKT':
    if args.graph_type == 'MHA':
        graph_model = MultiHeadAttention(args.edge_types, concept_num, args.emb_dim, args.attn_dim, dropout=args.dropout)
    elif args.graph_type == 'VAE':
        graph_model = VAE(args.emb_dim, args.vae_encoder_dim, args.edge_types, args.vae_decoder_dim, args.vae_decoder_dim, concept_num,
                          edge_type_num=args.edge_types, tau=args.temp, factor=args.factor, dropout=args.dropout, bias=args.bias)
        vae_loss = VAELoss(concept_num, edge_type_num=args.edge_types, prior=args.prior, var=args.var)
        if args.cuda:
            vae_loss = vae_loss.cuda()
    if args.cuda and args.graph_type in ['MHA', 'VAE']:
        graph_model = graph_model.cuda()

    model = GKT(concept_num, args.hid_dim, args.emb_dim, args.edge_types, args.graph_type, graph=graph, graph_model=graph_model,
                dropout=args.dropout, bias=args.bias, has_cuda=args.cuda)
elif args.model == 'DKT':
    model = DKT(res_len * concept_num, args.emb_dim, concept_num, dropout=args.dropout, bias=args.bias)
else:
    raise NotImplementedError(args.model + ' model is not implemented!')
kt_loss = KTLoss()


optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)


if args.load_dir:
    if args.model == 'DKT':
        model_file_name = 'DKT'
    elif args.model == 'GKT':
        model_file_name = 'GKT' + '-' + args.graph_type
    else:
        raise NotImplementedError(args.model + ' model is not implemented!')

    model_file = os.path.join(args.load_dir, model_file_name + '.pt')
    optimizer_file = os.path.join(save_dir, model_file_name + '-Optimizer.pt')
    scheduler_file = os.path.join(save_dir, model_file_name + '-Scheduler.pt')
    model.load_state_dict(torch.load(model_file))
    optimizer.load_state_dict(torch.load(optimizer_file))
    scheduler.load_state_dict(torch.load(scheduler_file))
    args.save_dir = False

# build optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)

if args.model == 'GKT' and args.prior:
    prior = np.array([0.91, 0.03, 0.03, 0.03])  # TODO: hard coded for now
    print("Using prior")
    print(prior)
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)
    if args.cuda:
        log_prior = log_prior.cuda()

if args.cuda:
    model = model.cuda()
    kt_loss = KTLoss()


def train(epoch, best_val_loss):
    t = time.time()
    loss_train = []
    kt_train = []
    vae_train = []
    auc_train = []
    acc_train = []
    if graph_model is not None:
        graph_model.train()
    model.train()
    for batch_idx, (features, questions, answers, user_ids) in enumerate(train_loader):
        t1 = time.time()
        if args.cuda:
            features, questions, answers = features.cuda(), questions.cuda(), answers.cuda()
        ec_list, rec_list, z_prob_list = None, None, None
        if args.model == 'GKT':
            pred_res, ec_list, rec_list, z_prob_list = model(features, questions)
        elif args.model == 'DKT':
            pred_res = model(features, questions)
        else:
            raise NotImplementedError(args.model + ' model is not implemented!')
        loss_kt, auc, acc = kt_loss(pred_res, answers)
        kt_train.append(float(loss_kt.cpu().detach().numpy()))
        if auc != -1 and acc != -1:
            auc_train.append(auc)
            acc_train.append(acc)

        if args.model == 'GKT' and args.graph_type == 'VAE':
            if args.prior:
                loss_vae = vae_loss(ec_list, rec_list, z_prob_list, log_prior=log_prior)
            else:
                loss_vae = vae_loss(ec_list, rec_list, z_prob_list)
                vae_train.append(float(loss_vae.cpu().detach().numpy()))
            print('batch idx: ', batch_idx, 'loss kt: ', loss_kt.item(), 'loss vae: ', loss_vae.item(), 'auc: ', auc, 'acc: ', acc, end=' ')
            loss = loss_kt + loss_vae
        else:
            loss = loss_kt
            print('batch idx: ', batch_idx, 'loss kt: ', loss_kt.item(), 'auc: ', auc, 'acc: ', acc, end=' ')
        loss_train.append(float(loss.cpu().detach().numpy()))
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        del loss
        print('cost time: ', str(time.time() - t1))

    loss_val = []
    kt_val = []
    vae_val = []
    auc_val = []
    acc_val = []

    if graph_model is not None:
        graph_model.eval()
    model.eval()
    with torch.no_grad():
        for batch_idx, (features, questions, answers, user_ids) in enumerate(valid_loader):
            if args.cuda:
                features, questions, answers = features.cuda(), questions.cuda(), answers.cuda()
            ec_list, rec_list, z_prob_list = None, None, None
            if args.model == 'GKT':
                pred_res, ec_list, rec_list, z_prob_list = model(features, questions)
            elif args.model == 'DKT':
                pred_res = model(features, questions)
            else:
                raise NotImplementedError(args.model + ' model is not implemented!')
            loss_kt, auc, acc = kt_loss(pred_res, answers)
            loss_kt = float(loss_kt.cpu().detach().numpy())
            kt_val.append(loss_kt)
            if auc != -1 and acc != -1:
                auc_val.append(auc)
                acc_val.append(acc)

            loss = loss_kt
            if args.model == 'GKT' and args.graph_type == 'VAE':
                loss_vae = vae_loss(ec_list, rec_list, z_prob_list)
                loss_vae = float(loss_vae.cpu().detach().numpy())
                vae_val.append(loss_vae)
                loss = loss_kt + loss_vae
            loss_val.append(loss)
            del loss
            
    # 에포크가 끝난 후 평균 값을 로깅
    avg_loss_val = sum(loss_val) / len(loss_val)
    avg_auc_val = sum(auc_val) / len(auc_val) if auc_val else 0
    avg_acc_val = sum(acc_val) / len(acc_val) if acc_val else 0
    
    mlflow.log_metric("Average Validation Loss", avg_loss_val, step=epoch)
    mlflow.log_metric("Average Validation AUC", avg_auc_val, step=epoch)
    mlflow.log_metric("Average Validation Accuracy", avg_acc_val, step=epoch)
                   
    if args.model == 'GKT' and args.graph_type == 'VAE':
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.10f}'.format(np.mean(loss_train)),
              'kt_train: {:.10f}'.format(np.mean(kt_train)),
              'vae_train: {:.10f}'.format(np.mean(vae_train)),
              'auc_train: {:.10f}'.format(np.mean(auc_train)),
              'acc_train: {:.10f}'.format(np.mean(acc_train)),
              'loss_val: {:.10f}'.format(np.mean(loss_val)),
              'kt_val: {:.10f}'.format(np.mean(kt_val)),
              'vae_val: {:.10f}'.format(np.mean(vae_val)),
              'auc_val: {:.10f}'.format(np.mean(auc_val)),
              'acc_val: {:.10f}'.format(np.mean(acc_val)),
              'time: {:.4f}s'.format(time.time() - t))
    else:
        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.10f}'.format(np.mean(loss_train)),
              'auc_train: {:.10f}'.format(np.mean(auc_train)),
              'acc_train: {:.10f}'.format(np.mean(acc_train)),
              'loss_val: {:.10f}'.format(np.mean(loss_val)),
              'auc_val: {:.10f}'.format(np.mean(auc_val)),
              'acc_val: {:.10f}'.format(np.mean(acc_val)),
              'time: {:.4f}s'.format(time.time() - t))
    if args.save_dir and np.mean(loss_val) < best_val_loss:
        print('Best model so far, saving...')

        save_pytorch_model_to_s3(model, BUCKET_NAME, f'{MODEL_PREFIX}model.pt')
        save_pytorch_model_to_s3(optimizer, BUCKET_NAME, f'{MODEL_PREFIX}optimizer.pt')
        save_pytorch_model_to_s3(scheduler, BUCKET_NAME, f'{MODEL_PREFIX}scheduler.pt')

        # torch.save(model.state_dict(), model_file)
        # torch.save(optimizer.state_dict(), optimizer_file)
        # torch.save(scheduler.state_dict(), scheduler_file)

        if args.model == 'GKT' and args.graph_type == 'VAE':
            print('Epoch: {:04d}'.format(epoch),
                  'loss_train: {:.10f}'.format(np.mean(loss_train)),
                  'kt_train: {:.10f}'.format(np.mean(kt_train)),
                  'vae_train: {:.10f}'.format(np.mean(vae_train)),
                  'auc_train: {:.10f}'.format(np.mean(auc_train)),
                  'acc_train: {:.10f}'.format(np.mean(acc_train)),
                  'loss_val: {:.10f}'.format(np.mean(loss_val)),
                  'kt_val: {:.10f}'.format(np.mean(kt_val)),
                  'vae_val: {:.10f}'.format(np.mean(vae_val)),
                  'auc_val: {:.10f}'.format(np.mean(auc_val)),
                  'acc_val: {:.10f}'.format(np.mean(acc_val)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
            del kt_train
            del vae_train
            del kt_val
            del vae_val
        else:
            print('Epoch: {:04d}'.format(epoch),
                  'loss_train: {:.10f}'.format(np.mean(loss_train)),
                  'auc_train: {:.10f}'.format(np.mean(auc_train)),
                  'acc_train: {:.10f}'.format(np.mean(acc_train)),
                  'loss_val: {:.10f}'.format(np.mean(loss_val)),
                  'auc_val: {:.10f}'.format(np.mean(auc_val)),
                  'acc_val: {:.10f}'.format(np.mean(acc_val)),
                  'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    res = np.mean(loss_val)
    del loss_train
    del auc_train
    del acc_train
    del loss_val
    del auc_val
    del acc_val
    gc.collect()
    if args.cuda:
        torch.cuda.empty_cache()
    return res

def download_model_from_s3(bucket_name, s3_key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=s3_key)
    model_data = response['Body'].read()  # S3에서 파일 데이터를 읽음
    return model_data


def test():
    loss_test = []
    kt_test = []
    vae_test = []
    auc_test = []
    acc_test = []

    bucket_name = 'jstest-4t'
    model_s3_key = 'model/model.pt'
    model_data = download_model_from_s3(bucket_name, model_s3_key)
    model.load_state_dict(torch.load(io.BytesIO(model_data)))
    model.eval()

    with torch.no_grad():
        for batch_idx, (features, questions, answers, user_ids) in enumerate(test_loader):
            
            if args.cuda:
                features, questions, answers = features.cuda(), questions.cuda(), answers.cuda()
            ec_list, rec_list, z_prob_list = None, None, None
            if args.model == 'GKT':
                pred_res, ec_list, rec_list, z_prob_list = model(features, questions)
            elif args.model == 'DKT':
                pred_res = model(features, questions)
            else:
                raise NotImplementedError(args.model + ' model is not implemented!')

            loss_kt, auc, acc = kt_loss(pred_res, answers)
            loss_kt = float(loss_kt.cpu().detach().numpy())
            if auc != -1 and acc != -1:
                auc_test.append(auc)
                acc_test.append(acc)
            kt_test.append(loss_kt)
            loss = loss_kt
            if args.model == 'GKT' and args.graph_type == 'VAE':
                loss_vae = vae_loss(ec_list, rec_list, z_prob_list)
                loss_vae = float(loss_vae.cpu().detach().numpy())
                vae_test.append(loss_vae)
                loss = loss_kt + loss_vae
            loss_test.append(loss)
            del loss

    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    if args.model == 'GKT' and args.graph_type == 'VAE':
        print('loss_test: {:.10f}'.format(np.mean(loss_test)),
              'kt_test: {:.10f}'.format(np.mean(kt_test)),
              'vae_test: {:.10f}'.format(np.mean(vae_test)),
              'auc_test: {:.10f}'.format(np.mean(auc_test)),
              'acc_test: {:.10f}'.format(np.mean(acc_test)))
    else:
        print('loss_test: {:.10f}'.format(np.mean(loss_test)),
              'auc_test: {:.10f}'.format(np.mean(auc_test)),
              'acc_test: {:.10f}'.format(np.mean(acc_test)))
    if args.save_dir:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)
        if args.model == 'GKT' and args.graph_type == 'VAE':
            print('loss_test: {:.10f}'.format(np.mean(loss_test)),
                  'kt_test: {:.10f}'.format(np.mean(kt_test)),
                  'vae_test: {:.10f}'.format(np.mean(vae_test)),
                  'auc_test: {:.10f}'.format(np.mean(auc_test)),
                  'acc_test: {:.10f}'.format(np.mean(acc_test)), file=log)
            del kt_test
            del vae_test
        else:
            print('loss_test: {:.10f}'.format(np.mean(loss_test)),
                  'auc_test: {:.10f}'.format(np.mean(auc_test)),
                  'acc_test: {:.10f}'.format(np.mean(acc_test)), file=log)
        log.flush()
        
    del loss_test
    del auc_test
    del acc_test
    gc.collect()
    if args.cuda:
        torch.cuda.empty_cache()

if args.test is False:
    # Train model
    print('start training!')
    t_total = time.time()
    best_val_loss = np.inf
    best_epoch = 0
    for epoch in range(args.epochs):
        val_loss = train(epoch, best_val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
    print("Optimization Finished!")
    print("Best Epoch: {:04d}".format(best_epoch))
    if args.save_dir:
        print("Best Epoch: {:04d}".format(best_epoch), file=log)
        log.flush()


mlflow.end_run()


object_key = f"{MODEL_PREFIX}saved_graph_model_and_concept_num.pkl"

# 그래프, 그래프 모델, concept_num을 pickle로 메모리에 저장
pickle_buffer = BytesIO()
pickle.dump((graph, graph_model, concept_num), pickle_buffer)
pickle_buffer.seek(0)  # 버퍼의 시작 위치로 이동

# S3에 업로드
s3_client.upload_fileobj(pickle_buffer, BUCKET_NAME, object_key)
print("S3에 파일 업로드 완료:", object_key)

test()
if log is not None:
    print(save_dir)
    log.close()