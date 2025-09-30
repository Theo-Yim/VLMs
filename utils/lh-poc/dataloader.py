import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image

class LHDataLoader:
    """
    K-LH-302 데이터셋을 위한 데이터로더 클래스
    메타 폴더의 이미지 파일과 라벨 폴더의 annotation을 연결합니다.
    """
    
    def __init__(self, data_root: str, type: str):
        """
        Args:
            data_root (str): 데이터셋 루트 폴더 경로
        """
        self.data_root = Path(data_root)
        self.annotation_path = self.data_root / f"lh-data-annotation-{type}"
        self.image_path = self.data_root / f"lh-data-image-{type}"
        
        # 데이터 인덱스 생성
        self.data_index = self._build_index()
        
        print(f"데이터로더 초기화 완료: {len(self.data_index)}개의 데이터 항목")
    
    def _build_index(self) -> List[Dict]:
        """
        메타 폴더를 탐색하여 데이터 인덱스를 생성합니다.
        
        Returns:
            List[Dict]: 각 데이터 항목의 정보를 담은 딕셔너리 리스트
        """
        index = []
        
        # 라벨 폴더에서 모든 JSON 파일 찾기
        annotation_json_files = list(self.annotation_path.glob("*.json"))
        
        print(f"라벨 폴더에서 {len(annotation_json_files)}개의 JSON 파일 발견")
        
        for annotation_file in annotation_json_files:
            try:
                # 이미지 파일명 추출 (JSON 파일명에서 .json 제거)
                label_id = annotation_file.stem
                
                if label_id:
                    # 해당하는 라벨 파일 경로
                    label_file = self.annotation_path / f"{label_id}.json"
                    
                    # 라벨 파일이 존재하는지 확인
                    if label_file.exists():
                        index.append({
                            'image_file': f"{self.image_path}/{label_id}.jpg",
                            'annotation_file': f"{self.annotation_path}/{label_id}.json",
                            'label_id': label_id,
                        })
                    else:
                        print(f"경고: 라벨 파일이 존재하지 않음 - {label_file}")
                
            except Exception as e:
                print(f"메타 파일 처리 중 오류: {annotation_file} - {e}")
        
        return index
    
    def __len__(self) -> int:
        """데이터셋 크기 반환"""
        return len(self.data_index)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        특정 인덱스의 데이터 항목을 반환합니다.
        
        Args:
            idx (int): 인덱스
            
        Returns:
            Dict: 데이터 항목 (메타 데이터와 라벨 데이터 포함)
        """
        if idx >= len(self.data_index):
            raise IndexError(f"인덱스 {idx}는 범위를 벗어남 (데이터셋 크기: {len(self.data_index)})")
        
        item = self.data_index[idx]
        
        # 메타 데이터 로드
        with open(item['annotation_file'], 'r', encoding='utf-8') as f:
            annotation_data = json.load(f)

        return {
            'index': idx,
            'image_file': item['image_file'],
            'annotation_file': item['annotation_file'],
            'label_id': item['label_id'],
            'annotation_data': annotation_data,
        }
    
    def get_by_label_id(self, label_id: str) -> Optional[Dict]:
        """
        라벨 ID로 데이터 항목을 찾습니다.
        
        Args:
            label_id (str): 라벨 ID
            
        Returns:
            Optional[Dict]: 데이터 항목 또는 None
        """
        for idx, item in enumerate(self.data_index):
            if item['label_id'] == label_id:
                return self.__getitem__(idx)
        return None
    
    def get_label_ids(self) -> List[str]:
        """모든 라벨 ID 리스트 반환"""
        return [item['label_id'] for item in self.data_index]
    
    def get_statistics(self) -> Dict:
        """데이터셋 통계 정보 반환"""
        stats = {
            'total_items': len(self.data_index),
            'unique_label_ids': len(set(self.get_label_ids())),
        }
        
        return stats

    def get_ground_truth(self) -> Dict:
        """데이터셋 그라운드 트루 정보 반환"""
        ground_truth = {}
        for idx, item in enumerate(self.data_index):
            item = self.__getitem__(idx)            
            metadata = item['annotation_data']['metadata']
            for key, value in metadata.items():
                if key == 'tags' or key =='하자내용':
                    continue
                if key not in ground_truth:
                    ground_truth[key] = set()
                if isinstance(value, list):
                    for v in value:
                        ground_truth[key].add(v)
                else:
                    ground_truth[key].add(value)
        for key, value in ground_truth.items():
            ground_truth[key] = {v:v for v in value}
        return ground_truth

def main():
    """사용 예시"""
    # 데이터 경로 설정
    data_root = "/mnt/nas2/users/sbchoi/kh-practices/lh-poc/"
    
    # 데이터로더 생성
    loader = LHDataLoader(data_root, type="train")
    
    # 통계 정보 출력
    stats = loader.get_statistics()
    print("\n=== 데이터셋 통계 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 첫 번째 데이터 항목 확인
    if len(loader) > 0:
        index = 0
        for item in loader:
            print("\n=== 첫 번째 데이터 항목 ===")
            print(f"이미지 파일: {item['image_file']}")
            print(f"라벨 ID: {item['label_id']}")
            print(f"라벨 파일: {item['annotation_file']}")
            print(f"라벨 데이터: {item['annotation_data']}")
            index += 1
            if index > 1:
                break
    
    # 특정 라벨 ID로 검색 예시
    if len(loader) > 0:
        label_ids = loader.get_label_ids()
        if label_ids:
            print("\n=== 라벨 ID로 검색 예시 ===")
            sample_label_id = label_ids[0]
            item = loader.get_by_label_id(sample_label_id)
            if item:
                print(f"라벨 ID {sample_label_id}의 이미지: {item['image_file']}")

    ground_truth = loader.get_ground_truth()
    print("\n=== 그라운드 트루 정보 ===")
    for key, value in ground_truth.items():
        print(f"{key}: {value}")

    if not os.path.exists("ground_truth_mapping.json"):
        json.dump(ground_truth, open("ground_truth_mapping.json", "w"), ensure_ascii=False, indent=4)
    else:
        print("ground_truth_mapping.json already exists, are you sure you want to overwrite it?")

if __name__ == "__main__":
    main()
