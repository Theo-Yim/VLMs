import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('plot/similarity_scores.csv')

# actual defect 컬럼 읽기
actual_defects = df['actual_material'].tolist()

print("실제 결함 목록:")
print(list(set(actual_defects)))
