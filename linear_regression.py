import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# 1. CSV 불러오기
df = pd.read_csv('csv_example.csv')

# 2. 입력과 출력 분리
X = df[['in1', 'in2', 'in3']].values
y = df['out'].values

# 3. 입력 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Ridge 회귀 학습
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)

# 5. 정규화된 가중치 추출
raw_weights = ridge.coef_
weights_normalized = raw_weights / np.sum(raw_weights)

print("정규화된 가중치:", weights_normalized)

# 6. 새 입력 예시
new_sample = np.array([[0.6, 0.3, 0.4]])

# 7. 예측값 계산 (정규화된 가중합 방식)
score = np.dot(weights_normalized, new_sample.T)
print("예측된 concentration score:", score)
