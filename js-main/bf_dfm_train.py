import pandas as pd
import boto3
from io import BytesIO
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import joblib
from tensorflow.keras.utils import register_keras_serializable
import pickle
import io
import os
import tempfile
import mlflow
from dotenv import load_dotenv

load_dotenv()
# AWS S3 설정
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
mlflow.set_tracking_uri(mlflow_tracking_uri)

s3_bucket_name = "result-4t"                      
bf_file_key = "mlflow/deepfm_mlflow/bf_merged_df.csv"              # S3에 저장된 파일 경로
MODEL_PREFIX = "mlflow/deepfm_mlflow/"
bf_model_key = "mlflow/deepfm_mlflow/bf_deepfm_nn.keras"

# S3 클라이언트 설정
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', "http://43.203.12.181:5000")
experiment_name = "new_experiment_name"
experiment_id = "745537454314509167"
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment(experiment_name)  

# S3에서 CSV 파일 로드 함수
def load_csv_from_s3(bucket, key):
    """S3에서 CSV 파일을 로드하여 DataFrame으로 반환"""
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(BytesIO(obj['Body'].read()))

# S3에서 데이터 로드
bf_merged_df = load_csv_from_s3(s3_bucket_name, bf_file_key)

bf_merged_df = bf_merged_df.dropna()

# 모델 초기화
feature_size = 196  # skill_0부터 skill_195까지의 수
embedding_size = 8  # 임베딩 차원

@register_keras_serializable()
class DeepFM(tf.keras.Model):
    def __init__(self, feature_size, embedding_size, num_classes, name=None, trainable=True, dtype=None):
        super(DeepFM, self).__init__(name=name, trainable=trainable, dtype=dtype)  # dtype 인자를 부모 클래스에 전달
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.num_classes = num_classes

        # Factorization Machine (FM) 부분
        self.embedding = tf.keras.layers.Embedding(input_dim=feature_size, output_dim=embedding_size)
        
        # DNN 부분
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1),  # 최종 출력을 스칼라 값으로
        ])
        
        # 출력 레이어
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='sigmoid')  # 다중 클래스

    def call(self, inputs):
        # Factorization Machine (FM) 계산
        embeddings = self.embedding(inputs)
        summed_embeddings = tf.reduce_sum(embeddings, axis=1)
        squared_sum_embeddings = tf.square(summed_embeddings)
        summed_square_embeddings = tf.reduce_sum(tf.square(embeddings), axis=1)
        fm_output = 0.5 * tf.subtract(squared_sum_embeddings, summed_square_embeddings)

        # DNN 계산
        dnn_output = self.dense_layers(embeddings)

        # DNN의 출력을 1D로 변환
        dnn_output = tf.reduce_sum(dnn_output, axis=1)  # (None, 1)로 변환

        # FM과 DNN의 출력을 결합
        final_output = fm_output + dnn_output  # (None, 1)로 일치
        return self.output_layer(final_output)

    def get_config(self):
        config = super(DeepFM, self).get_config()
        config.update({
            'feature_size': self.feature_size,
            'embedding_size': self.embedding_size,
            'num_classes': self.num_classes,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)  # config에서 모든 인자를 전달
    
@register_keras_serializable()
class HammingAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="hamming_accuracy", **kwargs):
        super(HammingAccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true를 float32로 캐스팅하여 y_pred와 타입을 맞춤
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.round(y_pred)  # 예측 결과를 0 또는 1로 반올림하여 이진 결과로 처리

        # 맞춘 레이블의 수 계산
        match = tf.reduce_sum(tf.cast(y_true == y_pred, tf.float32), axis=1)
        # 총 레이블 개수와 일치율 계산
        accuracy = match / tf.cast(tf.shape(y_true)[1], tf.float32)
        
        # 총 정확도와 카운트를 업데이트
        self.total.assign_add(tf.reduce_sum(accuracy))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.total / self.count

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

# 입력 피처와 타겟 분리
bf_X = bf_merged_df.drop('bf_skill', axis=1)
bf_y = bf_merged_df['bf_skill']

mlb = MultiLabelBinarizer()
bf_y_encoded = mlb.fit_transform(bf_y.str.split('|'))
object_key = f"{MODEL_PREFIX}bf_mlb.pkl"

# 그래프, 그래프 모델, concept_num을 pickle로 메모리에 저장
pickle_buffer = BytesIO()
pickle.dump(mlb, pickle_buffer)
pickle_buffer.seek(0)  # 버퍼의 시작 위치로 이동

# S3에 업로드
s3_client.upload_fileobj(pickle_buffer, s3_bucket_name, object_key)
print("S3에 파일 업로드 완료:", object_key)

# 데이터 나누기 (훈련:검증:테스트 = 70:15:15)
bf_X_train, bf_X_temp, bf_y_train, bf_y_temp = train_test_split(bf_X, bf_y_encoded, test_size=0.3, random_state=42)
bf_X_val, bf_X_test, bf_y_val, bf_y_test = train_test_split(bf_X_temp, bf_y_temp, test_size=0.5, random_state=42)

bf_num_classes = bf_y_encoded.shape[1]  # 클래스 수
bf_model = DeepFM(feature_size, embedding_size, bf_num_classes)
bf_model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=[HammingAccuracy()])

# MLflow Run 시작
with mlflow.start_run():
    # 하이퍼파라미터 로깅
    mlflow.log_params({
        "feature_size": feature_size,
        "embedding_size": embedding_size,
        "num_classes": bf_num_classes,
        "epochs": 20,
        "batch_size": 32
    })

    # 모델 학습 및 에포크별 메트릭 로깅
    epochs = 20
    for epoch in range(epochs):
        # 각 에포크별 학습
        history = bf_model.fit(
            bf_X_train, bf_y_train,
            validation_data=(bf_X_val, bf_y_val),
            epochs=1,  # 루프에서 하나의 에포크씩 학습
            batch_size=32,
            verbose=1
        )

        # 에포크별 평균 손실, AUC, 정확도 계산 및 로깅
        avg_loss_val = history.history['val_loss'][0]
        avg_acc_val = history.history['val_hamming_accuracy'][0]

        # MLflow에 메트릭 로깅
        mlflow.log_metric("Average Validation Loss", avg_loss_val, step=epoch)
        mlflow.log_metric("Average Validation Accuracy", avg_acc_val, step=epoch)

    # 모델을 임시 파일에 저장 후 S3 및 MLflow에 로깅
    with tempfile.NamedTemporaryFile(suffix=".keras") as tmp:
        tf.keras.models.save_model(bf_model, tmp.name)
        tmp.seek(0)

        # S3에 업로드
        s3_client.upload_file(tmp.name, s3_bucket_name, bf_model_key)
        print("모델이 S3에 성공적으로 저장되었습니다:", bf_model_key)

        # MLflow에 모델 아티팩트로 저장
        mlflow.log_artifact(tmp.name, artifact_path="model")