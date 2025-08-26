from spb_label import sdk
from spb_label.utils import SearchFilter
import requests
import os
from urllib.parse import urlparse
from pathlib import Path
import zipfile

source_tenant_id = "superbai-lh"
source_access_key = "0d48sngTavirJHirThOM1cJiRleNCrT1VaK48tUd"
source_project_name = "K-LH-302"

client = sdk.Client(project_name=source_project_name, team_name=source_tenant_id, access_key=source_access_key)
export = client.get_export(name="K-LH-302 2025-08-22 155843")

print("Export ID: {}".format(export.id))
print("Export Name: {}".format(export.name))
print("Export State: {}".format(export.state))
print("Export Download URL: {}".format(export.download_url))

def download_from_url(url, folder_path, filename=None):
    """
    URL에서 파일을 다운로드하여 지정된 폴더에 저장합니다.
    
    Args:
        url (str): 다운로드할 URL
        folder_path (str): 저장할 폴더 경로
        filename (str, optional): 저장할 파일명. None이면 URL에서 추출
    
    Returns:
        str: 다운로드된 파일의 전체 경로
    """
    try:
        # 폴더가 존재하지 않으면 생성
        os.makedirs(folder_path, exist_ok=True)
        
        # 파일명이 지정되지 않은 경우 URL에서 추출
        if filename is None:
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename:
                filename = "downloaded_file"
        
        # 전체 파일 경로
        file_path = os.path.join(folder_path, filename)
        
        print(f"다운로드 시작: {url}")
        print(f"저장 위치: {file_path}")
        
        # 파일 다운로드
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 파일 크기 확인 (가능한 경우)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as file:
            if total_size > 0:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        downloaded += len(chunk)
                        progress = (downloaded / total_size) * 100
                        print(f"\r진행률: {progress:.1f}%", end='', flush=True)
                print()  # 줄바꿈
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
        
        print(f"다운로드 완료: {file_path}")
        return file_path
        
    except requests.RequestException as e:
        print(f"다운로드 오류: {e}")
        return None
    except Exception as e:
        print(f"일반 오류: {e}")
        return None

# Export URL을 사용하여 다운로드 실행
if export.download_url:
    download_folder = "./lh-data"
    downloaded_file = download_from_url(export.download_url, download_folder, f"{export.name}_export.zip")
    if downloaded_file:
        print(f"파일이 성공적으로 다운로드되었습니다: {downloaded_file}")
    else:
        print("다운로드에 실패했습니다.")
else:
    print("다운로드 URL이 없습니다.")

# 다운로드된 zip 파일 압축 해제
if downloaded_file and os.path.exists(downloaded_file):
    try:
        print(f"압축 해제 시작: {downloaded_file}")
        with zipfile.ZipFile(downloaded_file, 'r') as zip_ref:
            extract_path = os.path.splitext(downloaded_file)[0]  # .zip 확장자 제거
            zip_ref.extractall(extract_path)
        print(f"압축 해제 완료. 압축 해제된 파일 위치: {extract_path}")
    except zipfile.BadZipFile:
        print("잘못된 zip 파일입니다.")
    except Exception as e:
        print(f"압축 해제 중 오류 발생: {e}")

# get single datahandle for image or video by its image id
image_download_url_link = "https://suite-civet-asset-prod-s3.s3-accelerate.amazonaws.com/tenants/superbai-lh/projects/1ee4665c-4575-49b5-b22e-96df7b683cef/K-LH-302.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAQEBLWAF4OSNUSZ3P%2F20250822%2Fap-northeast-2%2Fs3%2Faws4_request&X-Amz-Date=20250822T070725Z&X-Amz-SignedHeaders=host&X-Amz-Expires=3600&X-Amz-Security-Token=IQoJb3JpZ2luX2VjELf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDmFwLW5vcnRoZWFzdC0yIkcwRQIhAIGamUCHt3jQoBsqGxXClHMWB6LxvfK%2FCmDT0nnWvLR5AiB8EvxOpzGSeT%2BUtT2VNlgFtgBgn6Vh7M6LsE7%2Fn3gk0yqIBAgQEAIaDDAwODY4MTU1NDI5NiIM%2FF%2FuXDe4kdKaeOc2KuUDPYP78%2FXIrSf7QElgJxo8RLPGDEbAs59y4Zze2EniczQf974fvfkyLbm5V2huGUua2xgbjLX%2F%2Ft48b8ETC7BuuwKuETJces8qQdFkSpc10NEHswkFxXyVfUTgTFFIH6zYi8JsBJTTvYQ6RYjf%2F1OCZeIhhUAQfQzDa4fIioBAR%2BMT07Fo0VcCwIktMETxffQvtB0IvcYeaVdjFTPNW2DOk%2FyFaCGqlttjSZbOpWBFzRzeQY%2BIMqP0LxgYn81Txv9XvKxJN6SlHYl60NSyLi%2Fpwzyu6WKxi%2B5e9Y3%2F3zWvuGkI3vRFVSZnkk8EzloF6mjskxUQLIFP%2FGjcw3kaI9dC5GggTSQVG3bWxhsNwnY6mLPdcW8rxntU4DlvYk%2F7wwMKB3aJDKIPY837Vuy1oHkajsB4tciDqcJe%2B1k8lA8zExci3TBOKi41h2ASyAVzaWEoDMbAxOI4WwGj37w0fpDlDbx3%2BFXU7dnVtjGDWcKpkYQWCWn6%2BY7Gil1bed89XHx%2FL1wpCsRuINCbWAMLZlLF7StWuUtf%2BO6JNWZew0aE%2Fh56zS1DRR95azliCO%2FHbQQvxT0rPQt3zihHhUYy4hr6ro70rB4%2B6TB3P7aMP7xb9h2E7TD3SWcEQPIiZBcKJ1jaZpV%2BmbUwuKugxQY6pQFzrbXgBYbp4UvLfXCVnbXY6uofRMYDxv6UZlAaYmAS6LotBCstoHMthh5n6VBIQl98a5ukjQ406MHY2wCnK3IhxEpu%2F%2BINtrEGpfxrDgLzJsy8cp2F0r4fZn5%2FplUyt04pxl6buzp%2Bz72Ufw2vXY8NVhrUX1tz3fBAd%2BPCZbg13VcWL06owmPhzlsMg%2BY0l95B5aWl9WYEA3KAKak1jsWnImaTSpc%3D&X-Amz-Signature=2e3b0ce66a7ebdb84f70ab50aeadaf51ad21be10da0a0ca66824c9143bf8d32b"

# Export URL을 사용하여 다운로드 실행
if image_download_url_link:
    download_folder = "./lh-data-image"
    if os.path.exists(download_folder):
        print(f"이미 다운로드된 파일이 존재합니다: {download_folder}")
        exit()
    else:
        os.makedirs(download_folder, exist_ok=True)
    downloaded_file = download_from_url(image_download_url_link, download_folder, "image.zip")
    if downloaded_file:
        print(f"파일이 성공적으로 다운로드되었습니다: {downloaded_file}")
    else:
        print("다운로드에 실패했습니다.")
else:
    print("다운로드 URL이 없습니다.")

# 다운로드된 zip 파일 압축 해제
if downloaded_file and os.path.exists(downloaded_file):
    try:
        print(f"압축 해제 시작: {downloaded_file}")
        with zipfile.ZipFile(downloaded_file, 'r') as zip_ref:
            extract_path = os.path.splitext(downloaded_file)[0]  # .zip 확장자 제거
            zip_ref.extractall(extract_path)
        print(f"압축 해제 완료. 압축 해제된 파일 위치: {extract_path}")
    except zipfile.BadZipFile:
        print("잘못된 zip 파일입니다.")
    except Exception as e:
        print(f"압축 해제 중 오류 발생: {e}")