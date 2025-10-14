import torch
from torchvision import models, transforms
from PIL import Image

CLASSES = ["Scallop", "Roundfish", "Crab", "Whelk", "Skate", "Flatfish", "Eel"]

@torch.no_grad()
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load("benthic_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(image):
    img_tensor = transform(image).unsqueeze(0)
    outputs = model(img_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    conf, idx = torch.max(probs, 0)
    return CLASSES[idx.item()], conf.item()
