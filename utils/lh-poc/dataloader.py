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
    
    def __init__(self, data_root: str, image_root: str):
        """
        Args:
            data_root (str): 데이터셋 루트 폴더 경로
        """
        self.data_root = Path(data_root)
        self.meta_path = self.data_root / "meta"
        self.labels_path = self.data_root / "labels"
        self.project_file = self.data_root / "project.json"
        self.image_root = Path(image_root)
        
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
        
        # 메타 폴더에서 모든 JSON 파일 찾기
        meta_json_files = list(self.meta_path.glob("**/*.json"))
        
        print(f"메타 폴더에서 {len(meta_json_files)}개의 JSON 파일 발견")
        
        for meta_file in meta_json_files:
            try:
                # 메타 파일 읽기
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                
                # 이미지 파일명 추출 (JSON 파일명에서 .json 제거)
                image_filename = meta_file.stem
                
                # 라벨 ID 찾기 (메타 데이터에서 추출)
                label_id = self._extract_label_id(meta_data, meta_file)
                
                if label_id:
                    # 해당하는 라벨 파일 경로
                    label_file = self.labels_path / f"{label_id}.json"
                    
                    # 라벨 파일이 존재하는지 확인
                    if label_file.exists():
                        index.append({
                            'image_filename': image_filename,
                            'meta_file': str(meta_file),
                            'label_id': label_id,
                            'label_file': str(label_file),
                            'relative_path': str(meta_file.relative_to(self.meta_path))
                        })
                    else:
                        print(f"경고: 라벨 파일이 존재하지 않음 - {label_file}")
                
            except Exception as e:
                print(f"메타 파일 처리 중 오류: {meta_file} - {e}")
        
        return index
    
    def _extract_label_id(self, meta_data: Dict, meta_file: Path) -> Optional[str]:
        """
        메타 데이터에서 label_id를 추출합니다.
        
        Args:
            meta_data (Dict): 메타 JSON 데이터
            meta_file (Path): 메타 파일 경로
            
        Returns:
            Optional[str]: 라벨 ID 또는 None
        """
        # 먼저 메타 데이터에서 label_id 필드 찾기
        if 'label_id' in meta_data:
            return meta_data['label_id']
        
        # 다른 가능한 필드들 확인
        possible_fields = ['id', 'annotation_id', 'task_id', 'data_id']
        for field in possible_fields:
            if field in meta_data:
                return meta_data[field]
        
        # 메타 데이터에서 objects나 annotations 내부 확인
        if 'objects' in meta_data and isinstance(meta_data['objects'], list):
            for obj in meta_data['objects']:
                if 'id' in obj:
                    return obj['id']
        
        # 파일명에서 추출 시도 (파일명이 UUID 형태인 경우)
        filename_parts = meta_file.stem.split('_')
        for part in filename_parts:
            if len(part) == 36 and part.count('-') == 4:  # UUID 형태 체크
                return part
        
        print(f"경고: label_id를 찾을 수 없음 - {meta_file}")
        return None
    
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
        with open(item['meta_file'], 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        
        # 라벨 데이터 로드
        with open(item['label_file'], 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        
        return {
            'index': idx,
            'image_filename': item['image_filename'],
            'label_id': item['label_id'],
            'meta_file': item['meta_file'],
            'label_file': item['label_file'],
            'relative_path': item['relative_path'],
            'meta_data': meta_data,
            'label_data': label_data
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
            'meta_folders': set(),
            'label_files_available': 0
        }
        
        for item in self.data_index:
            # 메타 폴더 구조 분석
            relative_parts = Path(item['relative_path']).parts
            if len(relative_parts) > 0:
                stats['meta_folders'].add(relative_parts[0])
            
            # 라벨 파일 존재 여부 확인
            if os.path.exists(item['label_file']):
                stats['label_files_available'] += 1
        
        stats['meta_folders'] = list(stats['meta_folders'])
        
        return stats

def main():
    """사용 예시"""
    # 데이터 경로 설정
    parent_path = "/mnt/nas1/data/lh-poc"
    data_root = os.path.join(parent_path, "lh-data/K-LH-302 2025-08-22 155843_export")  
    image_root = os.path.join(parent_path, "lh-data-image/image/20250722")
    
    # 데이터로더 생성
    loader = LHDataLoader(data_root, image_root)
    
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
            print(f"이미지 파일명: {item['image_filename']}")
            print(f"라벨 ID: {item['label_id']}")
            print(f"메타 파일: {item['meta_file']}")
            print(f"라벨 파일: {item['label_file']}")
            print(f"데이터 키: {item['meta_data']['data_key']}")
            image_path = loader.image_root / item['meta_data']['data_key']
            image = Image.open(image_path)
            print(f"이미지 경로: {image_path}")
            print(f"이미지 크기: {image.size}")

            tags = [tag['name'] for tag in item['meta_data']['tags']]
            print(f"태그: {tags}")
            information = {}
            if 'categories' not in item['label_data']:
                print(f"경고: 라벨 데이터에 categories 필드가 없음 - {item['label_file']}")
                continue
            properties = item['label_data']['categories']['properties']
            for property in properties:
                if 'property_name' in property:
                    if 'value' in property:
                        information[property['property_name']] = property['value']
                    elif 'option_names' in property:
                        information[property['property_name']] = property['option_names']
            print(f"정보: {information}")
            
            # 라벨 데이터의 주요 정보 출력
            label_data = item['label_data']
            if 'objects' in label_data:
                print(f"어노테이션 객체 수: {len(label_data['objects'])}")
                if label_data['objects']:
                    obj = label_data['objects'][0]
                    print(f"첫 번째 객체 클래스: {obj.get('class_name', 'N/A')}")
            index += 1
    
    # 특정 라벨 ID로 검색 예시
    if len(loader) > 0:
        label_ids = loader.get_label_ids()
        if label_ids:
            print("\n=== 라벨 ID로 검색 예시 ===")
            sample_label_id = label_ids[0]
            item = loader.get_by_label_id(sample_label_id)
            if item:
                print(f"라벨 ID {sample_label_id}의 이미지: {item['image_filename']}")

if __name__ == "__main__":
    main()
