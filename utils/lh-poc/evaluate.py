import os
from sentence_transformers import SentenceTransformer, util
from dataloader import LHDataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import matplotlib.font_manager as fm
import numpy as np
import json
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import re
import argparse

def standardize_result(result_content):
    split_result = re.sub(r'```json.*?```', '', result_content)
    try:
        json_result = json.loads(split_result)
    except:
        json_result = {}
        candidates = ['"space":', '"defect_present":', '"defect_type":', '"material_part":', '"defect_content":', '"defect_score":']
        for line in split_result.split("\n"):
            line = line.strip().lower()
            if any(candidate in line for candidate in candidates):
                parts = line.split(":")
                key = parts[0].strip().strip('"').strip(',')
                value = parts[1].strip().strip('"').strip(',')
                json_result[key] = value                
    return json_result

def text_similarity(model, text1, text2, embedding_cache=None):
    """
    Calculate text similarity with embedding caching for performance optimization.
    If text contains '/', split and compare each part, returning maximum similarity.
    """
    if embedding_cache is None:
        embedding_cache = {}
    
    # Split texts if they contain '/'
    text1_parts = text1.split('/') if '/' in text1 else [text1]
    text2_parts = text2.split('/') if '/' in text2 else [text2]
    
    max_score = 0
    
    # Compare all combinations and find maximum similarity
    for t1 in text1_parts:
        for t2 in text2_parts:
            # Get or compute embedding for t1
            if t1 not in embedding_cache:
                embedding_cache[t1] = model.encode(t1, convert_to_tensor=True)
            emb1 = embedding_cache[t1]
            
            # Get or compute embedding for t2
            if t2 not in embedding_cache:
                embedding_cache[t2] = model.encode(t2, convert_to_tensor=True)
            emb2 = embedding_cache[t2]

            cosine_score = util.cos_sim(emb1, emb2).item()
            max_score = max(max_score, cosine_score)
    
    return max_score

def get_most_similar(text, candidates, model, embedding_cache):
    if not candidates:
        return None
        
    # 캐시된 임베딩 사용 또는 새로 계산
    if text not in embedding_cache:
        embedding_cache[text] = model.encode(text, convert_to_tensor=True)
    text_embedding = embedding_cache[text]
    
    max_similarity = -1
    most_similar = None
    
    for candidate in candidates:
        if candidate not in embedding_cache:
            embedding_cache[candidate] = model.encode(candidate, convert_to_tensor=True)
        candidate_embedding = embedding_cache[candidate]
        
        similarity = util.cos_sim(text_embedding, candidate_embedding).item()
        
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar = candidate
            
    return most_similar, text, max_similarity, embedding_cache

def create_overview_visualization(df, space_accuracy, material_accuracy, defect_accuracy, plot_dir):
    """전체 개요 시각화"""
    plt.figure(figsize=(20, 12))
    
    # 서브플롯 1: 카테고리별 정확도
    plt.subplot(2, 4, 1)
    categories = ['공간', '자재', '하자']
    accuracies = [space_accuracy, material_accuracy, defect_accuracy]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = plt.bar(categories, accuracies, color=colors, alpha=0.7)
    plt.title('카테고리별 정확도', fontsize=14, fontweight='bold')
    plt.ylabel('정확도')
    plt.ylim(0, 1)
    
    # 막대 위에 정확도 값 표시
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 서브플롯 2: 점수 분포
    plt.subplot(2, 4, 2)
    score_labels = ['정답', '오답', '예측없음']
    
    space_counts = [df['space_score'].value_counts().get(1, 0), 
                   df['space_score'].value_counts().get(0, 0), 
                   df['space_score'].value_counts().get(-1, 0)]
    material_counts = [df['material_score'].value_counts().get(1, 0), 
                      df['material_score'].value_counts().get(0, 0), 
                      df['material_score'].value_counts().get(-1, 0)]
    defect_counts = [df['defect_score'].value_counts().get(1, 0), 
                    df['defect_score'].value_counts().get(0, 0), 
                    df['defect_score'].value_counts().get(-1, 0)]
    
    x = np.arange(len(score_labels))
    width = 0.25
    
    plt.bar(x - width, space_counts, width, label='공간', color='#FF6B6B', alpha=0.7)
    plt.bar(x, material_counts, width, label='자재', color='#4ECDC4', alpha=0.7)
    plt.bar(x + width, defect_counts, width, label='하자', color='#45B7D1', alpha=0.7)
    
    plt.title('카테고리별 점수 분포', fontsize=14, fontweight='bold')
    plt.ylabel('샘플 수')
    plt.xlabel('점수 유형')
    plt.xticks(x, score_labels)
    plt.legend()
    
    # 서브플롯 3: 전체 성능 요약
    plt.subplot(2, 4, 3)
    total_correct = (df['space_score'] == 1).sum() + (df['material_score'] == 1).sum() + (df['defect_score'] == 1).sum()
    total_predictions = ((df['space_score'] != -1).sum() + 
                        (df['material_score'] != -1).sum() + 
                        (df['defect_score'] != -1).sum())
    overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    
    plt.pie([total_correct, total_predictions - total_correct], 
            labels=[f'정답\n{total_correct}개', f'오답\n{total_predictions - total_correct}개'],
            colors=['#96CEB4', '#FFEAA7'], autopct='%1.1f%%', startangle=90)
    plt.title(f'전체 정확도: {overall_accuracy:.3f}', fontsize=14, fontweight='bold')
    
    # 서브플롯 4: 카테고리별 예측 성공률
    plt.subplot(2, 4, 4)
    prediction_rates = [
        (df['space_score'] != -1).sum() / len(df) * 100,
        (df['material_score'] != -1).sum() / len(df) * 100,
        (df['defect_score'] != -1).sum() / len(df) * 100
    ]
    
    bars = plt.bar(categories, prediction_rates, color=colors, alpha=0.7)
    plt.title('카테고리별 예측 성공률', fontsize=14, fontweight='bold')
    plt.ylabel('예측 성공률 (%)')
    plt.ylim(0, 100)
    
    for bar, rate in zip(bars, prediction_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 서브플롯 5-7: 각 카테고리별 Top 정답 분포
    categories_data = [
        ('공간', 'ground_truth_space', 'prediction_space', 'space_score'),
        ('자재', 'ground_truth_material', 'prediction_material', 'material_score'),
        ('하자', 'ground_truth_defect', 'prediction_defect', 'defect_score')
    ]
    
    for i, (cat_name, gt_col, pred_col, score_col) in enumerate(categories_data[:3]):
        plt.subplot(2, 4, 5 + i)
        
        # 정답인 경우의 분포
        correct_data = df[df[score_col] == 1][gt_col].value_counts().head(10)
        if len(correct_data) > 0:
            plt.barh(range(len(correct_data)), correct_data.values, color=colors[i], alpha=0.7)
            plt.yticks(range(len(correct_data)), [label[:15] + '...' if len(label) > 15 else label for label in correct_data.index])
            plt.title(f'{cat_name} - 정답 Top 10', fontsize=12, fontweight='bold')
            plt.xlabel('빈도')
        else:
            plt.text(0.5, 0.5, '정답 데이터 없음', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'{cat_name} - 정답 분포', fontsize=12, fontweight='bold')
    
    # 서브플롯 8: 오답 분석 요약
    plt.subplot(2, 4, 8)
    error_summary = []
    for cat_name, gt_col, pred_col, score_col in categories_data:
        total_with_pred = (df[score_col] != -1).sum()
        error_count = (df[score_col] == 0).sum()
        error_rate = error_count / total_with_pred * 100 if total_with_pred > 0 else 0
        error_summary.append(error_rate)
    
    bars = plt.bar(categories, error_summary, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7)
    plt.title('카테고리별 오답률', fontsize=14, fontweight='bold')
    plt.ylabel('오답률 (%)')
    plt.ylim(0, max(error_summary) * 1.1 if error_summary else 100)
    
    for bar, rate in zip(bars, error_summary):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    overview_path = os.path.join(plot_dir, 'evaluation_overview.png')
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"전체 개요 시각화가 {overview_path}에 저장되었습니다.")


def create_category_detailed_analysis(df, plot_dir):
    """개별 카테고리별 상세 분석"""
    categories_data = [
        ('공간', 'ground_truth_space', 'prediction_space', 'space_score', '#FF6B6B'),
        ('자재', 'ground_truth_material', 'prediction_material', 'material_score', '#4ECDC4'),
        ('하자', 'ground_truth_defect', 'prediction_defect', 'defect_score', '#45B7D1')
    ]
    
    for cat_name, gt_col, pred_col, score_col, color in categories_data:
        plt.figure(figsize=(16, 10))
        
        # 예측이 있는 데이터만 필터링
        cat_data = df.copy()
        
        if len(cat_data) == 0:
            plt.text(0.5, 0.5, f'{cat_name} 카테고리에 예측 데이터가 없습니다.', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
            plt.title(f'{cat_name} 카테고리 상세 분석', fontsize=16, fontweight='bold')
            plt.savefig(os.path.join(plot_dir, f'{cat_name}_detailed_analysis.png'), dpi=300, bbox_inches='tight')
            plt.show()
            continue
        
        # 서브플롯 1: Ground Truth 분포
        plt.subplot(2, 3, 1)
        gt_counts = cat_data[gt_col].value_counts().head(15)
        plt.barh(range(len(gt_counts)), gt_counts.values, color=color, alpha=0.7)
        plt.yticks(range(len(gt_counts)), [label[:20] + '...' if len(label) > 20 else label for label in gt_counts.index])
        plt.title(f'{cat_name} - Ground Truth 분포 (Top 15)', fontweight='bold')
        plt.xlabel('빈도')
        
        # 서브플롯 2: 예측값 분포
        plt.subplot(2, 3, 2)
        pred_counts = cat_data[pred_col].value_counts().head(15)
        plt.barh(range(len(pred_counts)), pred_counts.values, color=color, alpha=0.7)
        plt.yticks(range(len(pred_counts)), [label[:20] + '...' if len(label) > 20 else label for label in pred_counts.index])
        plt.title(f'{cat_name} - 예측값 분포 (Top 15)', fontweight='bold')
        plt.xlabel('빈도')
        
        # 서브플롯 3: 정확도 분포
        plt.subplot(2, 3, 3)
        score_counts = cat_data[score_col].value_counts()
        labels = ['정답', '오답']
        values = [score_counts.get(1, 0), score_counts.get(0, 0)]
        plt.pie(values, labels=labels, autopct='%1.1f%%', 
               colors=[color, '#FFB6C1'], startangle=90)
        plt.title(f'{cat_name} - 정확도 분포', fontweight='bold')
        
        # 서브플롯 4: 정답 케이스 Top 10
        plt.subplot(2, 3, 4)
        correct_cases = cat_data[cat_data[score_col] == 1][gt_col].value_counts().head(10)
        if len(correct_cases) > 0:
            plt.barh(range(len(correct_cases)), correct_cases.values, color=color, alpha=0.7)
            plt.yticks(range(len(correct_cases)), [label[:15] + '...' if len(label) > 15 else label for label in correct_cases.index])
            plt.title(f'{cat_name} - 정답 케이스 Top 10', fontweight='bold')
            plt.xlabel('빈도')
        else:
            plt.text(0.5, 0.5, '정답 케이스 없음', ha='center', va='center', transform=plt.gca().transAxes)
        
        # 서브플롯 5: 오답 케이스 분석
        plt.subplot(2, 3, 5)
        wrong_cases = cat_data[cat_data[score_col] == 0]
        if len(wrong_cases) > 0:
            wrong_gt_counts = wrong_cases[gt_col].value_counts().head(10)
            plt.barh(range(len(wrong_gt_counts)), wrong_gt_counts.values, color='#FFB6C1', alpha=0.7)
            plt.yticks(range(len(wrong_gt_counts)), [label[:15] + '...' if len(label) > 15 else label for label in wrong_gt_counts.index])
            plt.title(f'{cat_name} - 오답 GT 분포 Top 10', fontweight='bold')
            plt.xlabel('빈도')
        else:
            plt.text(0.5, 0.5, '오답 케이스 없음', ha='center', va='center', transform=plt.gca().transAxes)
        
        # 서브플롯 6: 카테고리별 성능 메트릭
        plt.subplot(2, 3, 6)
        total_samples = len(cat_data)
        correct_samples = (cat_data[score_col] == 1).sum()
        wrong_samples = (cat_data[score_col] == 0).sum()
        accuracy = correct_samples / total_samples if total_samples > 0 else 0
        
        # 텍스트로 메트릭 표시
        plt.text(0.1, 0.8, f'정확도: {accuracy:.3f}', fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
        plt.text(0.1, 0.6, f'정답 수: {correct_samples}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.4, f'오답 수: {wrong_samples}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.2, f'전체 수: {total_samples}', fontsize=12, transform=plt.gca().transAxes)
        plt.axis('off')
        plt.title(f'{cat_name} - 성능 메트릭', fontweight='bold')
        
        plt.tight_layout()
        detailed_path = os.path.join(plot_dir, f'{cat_name}_detailed_analysis.png')
        plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"{cat_name} 카테고리 상세 분석이 {detailed_path}에 저장되었습니다.")


def create_confusion_matrices(df, plot_dir):
    """Confusion Matrix 생성"""
    categories_data = [
        ('공간', 'ground_truth_space', 'prediction_space', 'space_score', '#FF6B6B'),
        ('자재', 'ground_truth_material', 'prediction_material', 'material_score', '#4ECDC4'),
        ('하자', 'ground_truth_defect', 'prediction_defect', 'defect_score', '#45B7D1')
    ]
    
    plt.figure(figsize=(36, 6))
    
    for i, (cat_name, gt_col, pred_col, score_col, color) in enumerate(categories_data):
        plt.subplot(1, 3, i+1)
        
        # 예측이 있는 데이터만 필터링
        cat_data = df.copy()
        
        if len(cat_data) == 0:
            plt.text(0.5, 0.5, f'{cat_name}\n예측 데이터 없음', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title(f'{cat_name} Confusion Matrix', fontweight='bold')
            continue

        all_classes = cat_data[gt_col].value_counts().index.tolist()
        
        # 해당 클래스들만 필터링
        filtered_data = cat_data[cat_data[gt_col].isin(all_classes) & cat_data[pred_col].isin(all_classes)]
        
        if len(filtered_data) > 0:
            # Confusion matrix 계산
            cm = confusion_matrix(filtered_data[gt_col], filtered_data[pred_col], labels=all_classes)
            
            # 히트맵 생성
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=[label[:10] + '...' if len(label) > 10 else label for label in all_classes],
                       yticklabels=[label[:10] + '...' if len(label) > 10 else label for label in all_classes])
            plt.title(f'{cat_name} Confusion Matrix', fontweight='bold')
            plt.xlabel('예측값')
            plt.ylabel('실제값')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
        else:
            plt.text(0.5, 0.5, f'{cat_name}\n충분한 데이터 없음', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title(f'{cat_name} Confusion Matrix', fontweight='bold')
    
    plt.tight_layout()
    cm_path = os.path.join(plot_dir, 'confusion_matrices.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion Matrix가 {cm_path}에 저장되었습니다.")


def create_category_distribution_comparison(df, plot_dir):
    """카테고리별 분포 비교"""
    categories_data = [
        ('공간', 'ground_truth_space', 'prediction_space', 'space_score', '#FF6B6B'),
        ('자재', 'ground_truth_material', 'prediction_material', 'material_score', '#4ECDC4'),
        ('하자', 'ground_truth_defect', 'prediction_defect', 'defect_score', '#45B7D1')
    ]
    
    plt.figure(figsize=(18, 12))
    
    for i, (cat_name, gt_col, pred_col, score_col, color) in enumerate(categories_data):
        # Ground Truth vs Prediction 분포 비교
        plt.subplot(2, 3, i+1)
        
        # 예측이 있는 데이터만 필터링
        cat_data = df[df[score_col] != -1].copy()
        
        if len(cat_data) == 0:
            plt.text(0.5, 0.5, f'{cat_name}\n데이터 없음', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title(f'{cat_name} - GT vs 예측 분포', fontweight='bold')
            continue
        
        # 상위 10개 클래스 선택
        gt_top = cat_data[gt_col].value_counts().head(10)
        pred_top = cat_data[pred_col].value_counts().head(10)
        
        # 공통 클래스 및 각각의 상위 클래스 포함
        all_classes = list(set(list(gt_top.index) + list(pred_top.index)))[:12]  # 최대 12개
        
        gt_values = [cat_data[gt_col].value_counts().get(cls, 0) for cls in all_classes]
        pred_values = [cat_data[pred_col].value_counts().get(cls, 0) for cls in all_classes]
        
        x = np.arange(len(all_classes))
        width = 0.35
        
        plt.bar(x - width/2, gt_values, width, label='Ground Truth', color=color, alpha=0.7)
        plt.bar(x + width/2, pred_values, width, label='예측값', color=color, alpha=0.4)
        
        plt.title(f'{cat_name} - GT vs 예측 분포', fontweight='bold')
        plt.xlabel('클래스')
        plt.ylabel('빈도')
        plt.xticks(x, [label[:8] + '...' if len(label) > 8 else label for label in all_classes], rotation=45)
        plt.legend()
        
        # 개별 정확도별 분포
        plt.subplot(2, 3, i+4)
        
        # 정답/오답별 Ground Truth 분포
        correct_data = cat_data[cat_data[score_col] == 1]
        wrong_data = cat_data[cat_data[score_col] == 0]
        
        if len(correct_data) > 0 or len(wrong_data) > 0:
            correct_counts = correct_data[gt_col].value_counts().head(8)
            wrong_counts = wrong_data[gt_col].value_counts().head(8)
            
            all_classes_acc = list(set(list(correct_counts.index) + list(wrong_counts.index)))[:10]
            
            correct_values = [correct_counts.get(cls, 0) for cls in all_classes_acc]
            wrong_values = [wrong_counts.get(cls, 0) for cls in all_classes_acc]
            
            x = np.arange(len(all_classes_acc))
            width = 0.35
            
            plt.bar(x - width/2, correct_values, width, label='정답', color='#96CEB4', alpha=0.7)
            plt.bar(x + width/2, wrong_values, width, label='오답', color='#FFEAA7', alpha=0.7)
            
            plt.title(f'{cat_name} - 정답/오답별 GT 분포', fontweight='bold')
            plt.xlabel('Ground Truth 클래스')
            plt.ylabel('빈도')
            plt.xticks(x, [label[:8] + '...' if len(label) > 8 else label for label in all_classes_acc], rotation=45)
            plt.legend()
        else:
            plt.text(0.5, 0.5, '분석할 데이터 없음', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title(f'{cat_name} - 정답/오답별 분포', fontweight='bold')
    
    plt.tight_layout()
    dist_path = os.path.join(plot_dir, 'category_distribution_comparison.png')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"카테고리별 분포 비교가 {dist_path}에 저장되었습니다.")


if __name__ == "__main__":
    # 한국어 폰트 설정
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
    print(f"한국어 폰트 설정 완료: {font_prop.get_name()}")
        
    # 음수 표시 설정
    plt.rcParams['axes.unicode_minus'] = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/mnt/nas2/users/sbchoi/kh-practices/lh-poc/")
    parser.add_argument("--result_dir", type=str, default="/mnt/nas2/users/sbchoi/kh-practices/lh-poc/lh-data-result-test")
    parser.add_argument("--name", type=str, default="OpenGVLab/InternVL3_5-8B")
    args = parser.parse_args()

    data_root = args.data_root
    model = args.name.replace("/", "-")
    result_dir = os.path.join(args.result_dir, model)
    plot_dir = f"/mnt/nas2/users/sbchoi/kh-practices/lh-poc/plot/{model}"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    loader = LHDataLoader(data_root, type="test")

    ground_truth_mapping_path = "ground_truth_mapping.json"
    prediction_mapping_path = "prediction_mapping.json"

    with open(ground_truth_mapping_path, 'r', encoding='utf-8') as f:
        ground_truth_mapping = json.load(f)

    if os.path.exists(prediction_mapping_path):
        with open(prediction_mapping_path, 'r', encoding='utf-8') as f:
                prediction_mapping = json.load(f)
    else:
        prediction_mapping = {}

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # Create embedding cache dictionary for reuse
    embedding_cache = {}

    all_spaces = []
    all_materials = []
    all_defects = []

    predicted_spaces = []
    predicted_materials = []
    predicted_defects = []

    name = ['공간', '부위자재', '하자유형']

    index = 0
    for item in tqdm(loader):
        ground_truth = {}
        ground_truth = item['annotation_data']['metadata']
        for n in name:
            if ground_truth[n] not in ground_truth_mapping[n]:
                ground_truth_mapping[n][ground_truth[n]] = {'mapped': ground_truth[n]}

        space = ground_truth_mapping['공간'][ground_truth['공간']]['mapped']
        material = ground_truth_mapping['부위자재'][ground_truth['부위자재']]['mapped']
        defect = ground_truth_mapping['하자유형'][ground_truth['하자유형']]['mapped']

        all_spaces.append(space)
        all_materials.append(material)
        all_defects.append(defect)

        label_id = item['label_id']
        result_path = os.path.join(result_dir, f"{label_id}.txt")
        
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                result_content = f.read()
            prediction = standardize_result(result_content)

            if 'space' in prediction:
                predicted_spaces.append(prediction['space'])
            if 'material_part' in prediction:
                predicted_materials.append(prediction['material_part'])
            if 'defect_type' in prediction:
                predicted_defects.append(prediction['defect_type'])

    all_spaces = list(set(all_spaces))
    all_materials = list(set(all_materials)) 
    all_defects = list(set(all_defects))

    for space in set(predicted_spaces):
        if space not in prediction_mapping['space']:
            most_similar, text, max_similarity, embedding_cache = get_most_similar(space, all_spaces, model, embedding_cache)
            prediction_mapping['space'][space] = most_similar

    for material in set(predicted_materials):
        if material not in prediction_mapping['material_part']:
            most_similar, text, max_similarity, embedding_cache = get_most_similar(material, all_materials, model, embedding_cache)
            prediction_mapping['material_part'][material] = most_similar

    for defect in set(predicted_defects):
        if defect not in prediction_mapping['defect_type']:
            most_similar, text, max_similarity, embedding_cache = get_most_similar(defect, all_defects, model, embedding_cache)
            prediction_mapping['defect_type'][defect] = most_similar

    # mapping을 json으로 저장
    with open(prediction_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(prediction_mapping, f, ensure_ascii=False, indent=4, sort_keys=True)

    # 결과를 저장할 리스트들
    evaluation_results = []
    space_scores = []
    material_scores = []
    defect_scores = []
    
    correct_space = 0
    correct_material = 0
    correct_defect = 0
    total_samples = 0
    
    for item in tqdm(loader):
        ground_truth = {}
        ground_truth = item['annotation_data']['metadata']
        ground_truth_space = ground_truth_mapping['공간'][ground_truth['공간']]
        ground_truth_material = ground_truth_mapping['부위자재'][ground_truth['부위자재']]
        ground_truth_defect = ground_truth_mapping['하자유형'][ground_truth['하자유형']]
        label_id = item['label_id']
        result_path = os.path.join(result_dir, f"{label_id}.txt")
        
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                result_content = f.read()
            prediction = standardize_result(result_content)
            
            # 각 카테고리별 예측값과 정답 비교
            prediction_space = None
            prediction_material = None
            prediction_defect = None
            
            if 'space' in prediction:
                prediction_space = prediction_mapping['space'][prediction['space']]
            if 'material_part' in prediction:
                prediction_material = prediction_mapping['material_part'][prediction['material_part']]
            if 'defect_type' in prediction:
                prediction_defect = prediction_mapping['defect_type'][prediction['defect_type']]
            
            # 각 카테고리별 일치 점수 계산 (1: 일치, 0: 불일치, -1: 예측 없음)
            space_score = 1 if prediction_space == ground_truth_space else 0
            material_score = 1 if prediction_material == ground_truth_material else 0
            defect_score = 1 if prediction_defect == ground_truth_defect else 0
            
            # 정확도 카운팅 (예측이 있는 경우만)
            if prediction_space is not None:
                if prediction_space == ground_truth_space:
                    correct_space += 1
            if prediction_material is not None:
                if prediction_material == ground_truth_material:
                    correct_material += 1
            if prediction_defect is not None:
                if prediction_defect == ground_truth_defect:
                    correct_defect += 1
            
            # 결과 저장
            result_row = {
                'label_id': label_id,
                'ground_truth_space': ground_truth_space,
                'prediction_space': prediction_space,
                'space_score': space_score,
                'ground_truth_material': ground_truth_material,
                'prediction_material': prediction_material,
                'material_score': material_score,
                'ground_truth_defect': ground_truth_defect,
                'prediction_defect': prediction_defect,
                'defect_score': defect_score
            }
            evaluation_results.append(result_row)
            
            space_scores.append(space_score)
            material_scores.append(material_score)
            defect_scores.append(defect_score)
            
            total_samples += 1
    
    # CSV 파일로 저장
    df = pd.DataFrame(evaluation_results)
    csv_path = os.path.join(plot_dir, 'evaluation_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"평가 결과가 {csv_path}에 저장되었습니다.")
    
    # 정확도 계산 및 출력
    space_accuracy = correct_space / len([s for s in space_scores])
    material_accuracy = correct_material / len([s for s in material_scores])
    defect_accuracy = correct_defect / len([s for s in defect_scores])
    
    print(f"\n=== 평가 결과 ===")
    print(f"전체 샘플 수: {total_samples}")
    print(f"공간 정확도: {space_accuracy:.3f} ({correct_space}/{len([s for s in space_scores])})")
    print(f"자재 정확도: {material_accuracy:.3f} ({correct_material}/{len([s for s in material_scores])})")
    print(f"하자 정확도: {defect_accuracy:.3f} ({correct_defect}/{len([s for s in defect_scores])})")
    
    # 1. 전체 개요 시각화
    create_overview_visualization(df, space_accuracy, material_accuracy, defect_accuracy, plot_dir)
    
    # 2. 개별 카테고리별 상세 분석
    create_category_detailed_analysis(df, plot_dir)
    
    # 3. Confusion Matrix 생성
    create_confusion_matrices(df, plot_dir)
    
    # 4. 카테고리별 분포 비교
    create_category_distribution_comparison(df, plot_dir)
    
    # 카테고리별 상세 분석 저장
    category_summary = {
        'space': {
            'accuracy': space_accuracy,
            'correct': correct_space,
            'total_predictions': len([s for s in space_scores if s != -1]),
            'no_prediction': space_scores.count(-1)
        },
        'material': {
            'accuracy': material_accuracy,
            'correct': correct_material,
            'total_predictions': len([s for s in material_scores if s != -1]),
            'no_prediction': material_scores.count(-1)
        },
        'defect': {
            'accuracy': defect_accuracy,
            'correct': correct_defect,
            'total_predictions': len([s for s in defect_scores if s != -1]),
            'no_prediction': defect_scores.count(-1)
        }
    }
    
    summary_path = os.path.join(plot_dir, 'evaluation_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(category_summary, f, ensure_ascii=False, indent=4)
    print(f"평가 요약이 {summary_path}에 저장되었습니다.")
