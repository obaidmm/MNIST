import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN, test_model

# Load Model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(DEVICE)
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()

# Load Test Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Test Model
test_model(model, test_loader, DEVICE)
