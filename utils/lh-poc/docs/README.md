# K-LH-302 데이터셋 데이터로더

이 프로젝트는 K-LH-302 건설 하자 관리 데이터셋을 위한 데이터로더와 다운로드 도구를 제공합니다.

## 📁 프로젝트 구조

```
lh-poc/
├── download_lh.py          # 데이터셋 다운로드 및 압축 해제 스크립트
├── dataloader.py          # 메타 데이터와 라벨 데이터를 연결하는 데이터로더
├── lh-data/               # 다운로드된 데이터셋 저장 폴더
│   └── K-LH-302 2025-08-22 155250_export/
│       ├── meta/          # 이미지 메타데이터 (폴더 구조별 정리)
│       ├── labels/        # 어노테이션 라벨 데이터 (UUID별 정리)
│       └── project.json   # 프로젝트 설정 파일
└── README.md
```

## 🔄 데이터 플로우

1. **다운로드**: `download_lh.py`로 SuperB AI Suite에서 데이터셋 다운로드
2. **압축 해제**: ZIP 파일을 자동으로 압축 해제
3. **인덱싱**: `LHDataLoader`가 메타 폴더의 파일들을 스캔하여 라벨과 매칭
4. **데이터 접근**: 인덱스를 통해 메타데이터와 라벨 데이터를 쉽게 접근

## 🚀 사용법

### 1. 데이터셋 다운로드

```python
python download_lh.py
```

이 스크립트는:
- SuperB AI Suite에서 export 데이터를 다운로드
- ZIP 파일을 자동으로 압축 해제
- `lh-data/` 폴더에 데이터 저장

### 2. 데이터로더 사용

```python
from dataloader import LHDataLoader

# 데이터로더 초기화
data_root = "./lh-data/K-LH-302 2025-08-22 155250_export"
loader = LHDataLoader(data_root)

# 데이터셋 크기 확인
print(f"총 데이터 개수: {len(loader)}")

# 첫 번째 데이터 항목 가져오기
item = loader[0]
print(f"이미지 파일명: {item['image_filename']}")
print(f"라벨 ID: {item['label_id']}")

# 라벨 ID로 특정 데이터 검색
specific_item = loader.get_by_label_id("4a398075-4d31-43a2-9981-a5673fe3ffdb")

# 모든 데이터 순회
for item in loader:
    meta_data = item['meta_data']
    label_data = item['label_data']
    # 데이터 처리 로직...
```

### 3. 데이터 구조

**메타 데이터 (Meta Data):**
- 경로: `meta/날짜/빌딩번호/보수전|보수후/이미지파일명.json`
- 내용: 이미지 파일 정보, data_key, 기타 메타 정보

**라벨 데이터 (Label Data):**
- 경로: `labels/{UUID}.json`
- 내용: 하자 어노테이션 정보, 바운딩 박스, 클래스 정보, 속성값

**반환되는 데이터 항목:**
```python
{
    'index': 0,                    # 데이터 인덱스
    'image_filename': '3-106.jpg', # 이미지 파일명
    'label_id': 'uuid-string',     # 라벨 ID (UUID)
    'meta_file': '/path/to/meta.json',   # 메타 파일 경로
    'label_file': '/path/to/label.json', # 라벨 파일 경로
    'relative_path': 'relative/path',    # 상대 경로
    'meta_data': {...},            # 메타 JSON 데이터
    'label_data': {...}            # 라벨 JSON 데이터
}
```

## 📊 데이터셋 정보

### 하자 관리 데이터셋 특징

- **도메인**: 건설 하자 관리
- **데이터 타입**: 이미지 + 바운딩 박스 어노테이션
- **클래스**: "하자" (defect detection)
- **속성 정보**:
  - 하자진행상태 (보수완료, 공사완료, 대금지급요청 등)
  - 공간 (침실, 거실 등)
  - 부위자재 (걸레받이, 벽지 등)
  - 하자구분 (시공사하자 등)
  - 하자유형 (파손, 균열 등)
  - 하자내용 (상세 설명)

### 어노테이션 구조

```json
{
    "objects": [
        {
            "id": "object-uuid",
            "class_id": "class-uuid", 
            "class_name": "하자",
            "annotation_type": "box",
            "annotation": {
                "coord": {
                    "x": 40.08,
                    "y": 145.55,
                    "width": 160.86,
                    "height": 95.32
                }
            }
        }
    ],
    "categories": {
        "properties": [
            {
                "property_name": "하자진행상태",
                "option_names": ["공사완료"]
            },
            {
                "property_name": "공간",
                "value": "침실3"
            }
        ]
    }
}
```

## 🛠️ 주요 기능

### LHDataLoader 클래스

- **자동 인덱싱**: 메타 폴더 구조를 자동으로 탐색하여 라벨과 매칭
- **효율적 접근**: 인덱스 기반으로 빠른 데이터 접근
- **유연한 검색**: 라벨 ID로 특정 데이터 검색 가능
- **통계 정보**: 데이터셋 통계 및 구조 분석 제공
- **오류 처리**: 파일 누락이나 형식 오류에 대한 안전한 처리

### 다운로드 도구

- **스트리밍 다운로드**: 대용량 파일도 메모리 효율적으로 처리
- **진행률 표시**: 다운로드 진행 상황 실시간 표시
- **자동 압축 해제**: ZIP 파일 자동 압축 해제
- **오류 처리**: 네트워크 오류 및 파일 오류 처리

## 📈 사용 예시

### 통계 정보 확인

```python
# 데이터셋 통계 확인
stats = loader.get_statistics()
print("=== 데이터셋 통계 ===")
for key, value in stats.items():
    print(f"{key}: {value}")
```

### 하자 어노테이션 분석

```python
# 모든 하자 객체 정보 수집
defect_info = []
for item in loader:
    label_data = item['label_data']
    if 'objects' in label_data:
        for obj in label_data['objects']:
            if obj.get('class_name') == '하자':
                defect_info.append({
                    'image': item['image_filename'],
                    'bbox': obj['annotation']['coord'],
                    'properties': label_data.get('categories', {}).get('properties', [])
                })

print(f"총 하자 객체 수: {len(defect_info)}")
```

## 📋 요구사항

- Python 3.7+
- requests (다운로드용)
- spb_label (SuperB AI SDK) -> pip install --upgrade superb-ai-label
- pathlib, json, os (내장 모듈)

## 🔧 설정

`download_lh.py`에서 다음 정보를 설정하세요:

```python
source_tenant_id = "superbai-lh"
source_access_key = "your-access-key"
source_project_name = "K-LH-302"
```

## 🐛 문제 해결

### 일반적인 문제

1. **파일 경로 오류**: 한글 파일명이나 공백이 포함된 경로로 인한 문제
2. **권한 오류**: 다운로드 폴더 생성 권한 확인
3. **네트워크 오류**: 인터넷 연결 및 방화벽 설정 확인
4. **메모리 부족**: 대용량 파일 처리시 스트리밍 방식 사용

### 디버깅 팁

```python
# 메타 파일과 라벨 파일 매칭 확인
for item in loader.data_index[:5]:  # 처음 5개만 확인
    print(f"메타: {item['meta_file']}")
    print(f"라벨: {item['label_file']}")
    print(f"존재여부: {os.path.exists(item['label_file'])}")
    print("---")
```

## 📞 지원

문제가 발생하거나 기능 개선 제안이 있으시면 이슈를 등록해주세요.
