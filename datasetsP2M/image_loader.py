from PIL import Image
from torchvision import transforms

def load_image_tensor(path, image_size=(224, 224), device='cpu'):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor