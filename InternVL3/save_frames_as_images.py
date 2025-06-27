import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def denormalize_tensor(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """정규화된 텐서를 원본 이미지로 역변환"""
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)

    # 정규화 해제
    denorm_tensor = tensor * std + mean
    # [0, 1] 범위로 클램핑
    denorm_tensor = torch.clamp(denorm_tensor, 0, 1)

    return denorm_tensor


def save_frames_as_images(pixel_values, output_dir):
    """텐서를 개별 이미지 파일로 저장"""
    from torchvision.utils import save_image
    import os

    os.makedirs(output_dir, exist_ok=True)

    # 정규화 해제
    denorm_frames = denormalize_tensor(pixel_values)

    # 각 프레임을 이미지로 저장
    for i, frame in enumerate(denorm_frames):
        save_image(frame, f"{output_dir}/frame_{i:03d}.png")


if __name__ == "__main__":
    # 사용 예시
    pixel_values = None
    save_frames_as_images(pixel_values, "output_frames")
