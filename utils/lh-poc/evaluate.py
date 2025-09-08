import os
from sentence_transformers import SentenceTransformer, util
from dataloader import LHDataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import matplotlib.font_manager as fm
from name import MATERIAL_CLASS, SPACE_CLASS, DEFECT_CLASS
import numpy as np

def standardize_result(result_content):
    split_result = result_content.split("\n")
    result_dict = {}
    for line in split_result:
        if line.startswith("space:"):
            result_dict["space"] = line.split(":")[1].strip()
        elif line.startswith("defect_present:"):
            result_dict["defect_present"] = line.split(":")[1].strip()
        elif line.startswith("defect_type:"):
            result_dict["defect_type"] = line.split(":")[1].strip()
        elif line.startswith("defectiveness_score:"):
            result_dict["defectiveness_score"] = line.split(":")[1].strip()
        elif line.startswith("material_part:"):
            result_dict["material_part"] = line.split(":")[1].strip()
        elif line.startswith("location_in_image:"):
            result_dict["location_in_image"] = line.split(":")[1].strip()
    return result_dict

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


if __name__ == "__main__":
    # Example usage
    font_path = '/usr/share/fonts/nanum/NanumGothicBold.ttf'
    font_name = fm.FontProperties(fname=font_path).get_name()
    
    plt.rcParams["font.family"] = font_name

    data_root = "/mnt/nas2/users/sbchoi/kh-practices/lh-poc/lh-data/K-LH-302 2025-08-22 155843_export"
    image_root = "/mnt/nas2/users/sbchoi/kh-practices/lh-poc/lh-data-image/image/20250722"
    result_dir = "/mnt/nas2/users/sbchoi/kh-practices/lh-poc/lh-data-result"
    plot_dir = "/mnt/nas2/users/sbchoi/kh-practices/lh-poc/plot"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    loader = LHDataLoader(data_root, image_root)
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # Create embedding cache dictionary for reuse
    embedding_cache = {}
    
    # Lists to store similarity scores and categories
    space_similarities = []
    material_similarities = []
    defect_similarities = []
    label_ids = []
    
    # Lists to store actual and predicted categories
    actual_spaces = []
    predicted_spaces = []
    actual_materials = []
    predicted_materials = []
    actual_defects = []
    predicted_defects = []
    
    index = 0
    for item in tqdm(loader):
        ground_truth = {}
        if 'categories' not in item['label_data']:
            print(f"경고: 라벨 데이터에 categories 필드가 없음 - {item['label_file']}")
            continue
        properties = item['label_data']['categories']['properties']
        for property in properties:
            if 'property_name' in property:
                if 'value' in property:
                    ground_truth[property['property_name']] = property['value']
                elif 'option_names' in property:
                    ground_truth[property['property_name']] = property['option_names']
        
        label_id = item['label_id']
        result_path = os.path.join(result_dir, f"{label_id}.txt")
        
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                result_content = f.read()
            prediction = standardize_result(result_content)
            
            # Calculate similarities and store categories
            if 'space' in prediction:
                if ground_truth['공간'] not in SPACE_CLASS:
                    print(f"경고: 공간 라벨 데이터에 없음 - {ground_truth['공간']}")
                    continue
                gt = SPACE_CLASS[ground_truth['공간']]
                pred = prediction['space']
                space_similarity = text_similarity(model, gt, pred, embedding_cache)
                actual_spaces.append(gt)
                predicted_spaces.append(pred)
            else:
                space_similarity = 0
                actual_spaces.append(ground_truth['공간'])
                predicted_spaces.append('unknown')
                
            if 'material_part' in prediction:
                if ground_truth['부위자재'] not in MATERIAL_CLASS:
                    print(f"경고: 부위자재 라벨 데이터에 없음 - {ground_truth['부위자재']}")
                    continue
                gt = MATERIAL_CLASS[ground_truth['부위자재']]
                pred = prediction['material_part']
                defect_present_similarity = text_similarity(model, gt, pred, embedding_cache)
                actual_materials.append(gt)
                predicted_materials.append(pred)
            else:
                defect_present_similarity = 0
                actual_materials.append(ground_truth['부위자재'])
                predicted_materials.append('unknown')
                
            if 'defect_type' in prediction:
                if ground_truth['하자유형'] not in DEFECT_CLASS:
                    print(f"경고: 하자유형 라벨 데이터에 없음 - {ground_truth['하자유형']}")
                    continue
                gt = DEFECT_CLASS[ground_truth['하자유형']]
                pred = prediction['defect_type']
                defect_type_similarity = text_similarity(model, gt, pred, embedding_cache)
                actual_defects.append(gt)
                predicted_defects.append(pred)
            else:
                defect_type_similarity = 0
                actual_defects.append(ground_truth['하자유형'])
                predicted_defects.append('unknown')
            
            # Store scores and label_id
            space_similarities.append(space_similarity)
            material_similarities.append(defect_present_similarity)
            defect_similarities.append(defect_type_similarity)
            label_ids.append(label_id)
        
    # Create figure with subplots for similarity scores
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Box plot of similarities
    data = [space_similarities, material_similarities, defect_similarities]
    ax1.boxplot(data, labels=['Space', 'Material', 'Defect Type'])
    
    # Add mean points
    means = [np.mean(space_similarities), np.mean(material_similarities), np.mean(defect_similarities)]
    ax1.plot([1, 2, 3], means, 'rd', label='Mean')
    
    ax1.set_title('Distribution of Similarity Scores')
    ax1.set_ylabel('Similarity Score')
    ax1.legend()
    
    # Plot 2: Scatter plot of individual scores
    x = range(len(label_ids))
    ax2.scatter(x, space_similarities, label='Space', marker='o')
    ax2.scatter(x, material_similarities, label='Material', marker='s') 
    ax2.scatter(x, defect_similarities, label='Defect Type', marker='^')
    ax2.set_title('Individual Similarity Scores')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Similarity Score')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'similarity_analysis.png'))
    plt.close()
    
    # 각 클래스별 빈도수 시각화
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
    
    # 공간 빈도수
    space_counts = pd.Series(actual_spaces).value_counts()
    space_counts.plot(kind='bar', ax=ax1)
    ax1.set_title('공간 분포')
    ax1.set_xlabel('공간 종류')
    ax1.set_ylabel('빈도수')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 부위자재 빈도수 
    material_counts = pd.Series(actual_materials).value_counts()
    material_counts.plot(kind='bar', ax=ax2)
    ax2.set_title('부위자재 분포')
    ax2.set_xlabel('부위자재 종류')
    ax2.set_ylabel('빈도수')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 하자유형 빈도수
    defect_counts = pd.Series(actual_defects).value_counts()
    defect_counts.plot(kind='bar', ax=ax3)
    ax3.set_title('하자유형 분포')
    ax3.set_xlabel('하자유형 종류') 
    ax3.set_ylabel('빈도수')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'class_distributions.png'))
    plt.close()
    
    # Category별 평균 유사도 계산 및 시각화
    # 데이터프레임 생성
    results_df = pd.DataFrame({
        'label_id': label_ids,
        'space_similarity': space_similarities,
        'material_similarity': material_similarities,
        'defect_similarity': defect_similarities,
        'actual_space': actual_spaces,
        'predicted_space': predicted_spaces,
        'actual_material': actual_materials,
        'predicted_material': predicted_materials,
        'actual_defect': actual_defects,
        'predicted_defect': predicted_defects
    })
    
    # 각 groundtruth category별 평균 유사도 계산
    space_mean_by_category = results_df.groupby('actual_space')['space_similarity'].mean().sort_values(ascending=False)
    material_mean_by_category = results_df.groupby('actual_material')['material_similarity'].mean().sort_values(ascending=False)
    defect_mean_by_category = results_df.groupby('actual_defect')['defect_similarity'].mean().sort_values(ascending=False)
    
    # Category별 평균 유사도 시각화
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(30, 20))  # 세로 크기 증가
    
    # 공간별 평균 유사도
    bars1 = ax1.bar(range(len(space_mean_by_category)), space_mean_by_category.values, color='skyblue')
    ax1.set_title('공간별 평균 유사도')
    ax1.set_xlabel('공간 종류')
    ax1.set_ylabel('평균 유사도')
    ax1.set_xticks(range(len(space_mean_by_category)))
    ax1.set_xticklabels(space_mean_by_category.index, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 값 표시
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 부위자재별 평균 유사도
    bars2 = ax2.bar(range(len(material_mean_by_category)), material_mean_by_category.values, color='lightgreen')
    ax2.set_title('부위자재별 평균 유사도')
    ax2.set_xlabel('부위자재 종류')
    ax2.set_ylabel('평균 유사도')
    ax2.set_xticks(range(len(material_mean_by_category)))
    ax2.set_xticklabels(material_mean_by_category.index, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 값 표시
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 하자유형별 평균 유사도
    bars3 = ax3.bar(range(len(defect_mean_by_category)), defect_mean_by_category.values, color='lightcoral')
    ax3.set_title('하자유형별 평균 유사도')
    ax3.set_xlabel('하자유형 종류')
    ax3.set_ylabel('평균 유사도')
    ax3.set_xticks(range(len(defect_mean_by_category)))
    ax3.set_xticklabels(defect_mean_by_category.index, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 값 표시
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout(h_pad=1.5)  # 서브플롯 간 간격 증가
    plt.savefig(os.path.join(plot_dir, 'mean_similarity_by_category.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Category별 평균 유사도 요약 출력
    print("\n=== 공간별 평균 유사도 ===")
    for category, mean_sim in space_mean_by_category.items():
        print(f"{category}: {mean_sim:.4f}")
    
    print("\n=== 부위자재별 평균 유사도 ===")
    for category, mean_sim in material_mean_by_category.items():
        print(f"{category}: {mean_sim:.4f}")
    
    print("\n=== 하자유형별 평균 유사도 ===")
    for category, mean_sim in defect_mean_by_category.items():
        print(f"{category}: {mean_sim:.4f}")

    
    # Save numerical results to CSV
    results_df.to_csv(os.path.join(plot_dir, 'similarity_scores.csv'), index=False)
    
    # Save category별 평균 유사도 결과도 CSV로 저장
    category_means = pd.DataFrame({
        'category_type': ['공간'] * len(space_mean_by_category) + 
                        ['부위자재'] * len(material_mean_by_category) + 
                        ['하자유형'] * len(defect_mean_by_category),
        'category_name': list(space_mean_by_category.index) + 
                        list(material_mean_by_category.index) + 
                        list(defect_mean_by_category.index),
        'mean_similarity': list(space_mean_by_category.values) + 
                          list(material_mean_by_category.values) + 
                          list(defect_mean_by_category.values)
    })
    category_means.to_csv(os.path.join(plot_dir, 'category_mean_similarities.csv'), index=False)