from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    preprocess_1 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    preprocess_2 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
    ])
    
    preprocess_3 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.ToTensor(),
    ])
    
    image_path = "./exercise_data/dog_img.png"
    image = Image.open(image_path)
    
    processed_image = preprocess_1(image)
    plt.imshow(processed_image.permute(1, 2, 0))  # チャンネル次元を最後に移動
    plt.show()
    
    processed_image = preprocess_2(image)
    plt.imshow(processed_image.permute(1, 2, 0))  # チャンネル次元を最後に移動
    plt.show()
    
    processed_image = preprocess_3(image)
    plt.imshow(processed_image.permute(1, 2, 0))  # チャンネル次元を最後に移動
    plt.show()