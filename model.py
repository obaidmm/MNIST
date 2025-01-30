import torch
from torch import nn
from torch.nn import functional as F

# 🚀 CONFIGURATION VARIABLES
EPOCH_BREAK_ACCURACY = 0.995  # Stop training if accuracy reaches 99.5%
TEST_BATCH_SIZE = 1000        # Number of images per test batch

# 🔹 Updated CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 🔹 First Convolutional Layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  
        self.bn1 = nn.BatchNorm2d(32)  # Normalization speeds up training
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 🔹 Extra Convolutional Layer for deeper learning
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 🔹 Dropout layers to reduce overfitting
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # 🔹 Dynamically Compute Fully Connected Layer Input Size
        self._to_linear = None  # Placeholder
        self._compute_fc_input_size()

        # 🔹 Fully Connected Layers
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def _compute_fc_input_size(self):
        """ Pass a fake input through conv layers to determine FC layer input size """
        with torch.no_grad():
            sample_input = torch.randn(1, 1, 28, 28)  # MNIST Image size
            output = self._forward_conv(sample_input)
            self._to_linear = output.view(1, -1).shape[1]  # Compute flattened size

    def _forward_conv(self, x):
        """ Forward pass through convolutional layers (helper function) """
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)  # Pooling after third conv layer
        return x

    def forward(self, x):
        """Defines how data flows through the model"""
        x = self._forward_conv(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)  # Log softmax activation for classification

# 🏋️‍♂️ Training Function
def train_model(model, device, data_loader, loss_func, optimizer, num_epochs):
    """
    Trains a model using the given data loader, loss function, and optimizer.

    Args:
        model: PyTorch model (CNN in this case)
        device: Training device (CPU/GPU)
        data_loader: Training dataset loader
        loss_func: Loss function (e.g., CrossEntropyLoss)
        optimizer: Optimization algorithm (e.g., Adam)
        num_epochs: Number of epochs to train

    Returns:
        train_loss, train_acc: Lists of loss and accuracy per epoch
    """
    train_loss, train_acc = [], []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)  # Get highest probability class
            correct += predicted.eq(labels).sum().item()  # Count correct predictions
            total += labels.size(0)  # Track total images processed

        # Calculate loss & accuracy
        epoch_loss = running_loss / len(data_loader)
        epoch_acc = correct / total
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Stop training early if accuracy is high enough
        if epoch_acc >= EPOCH_BREAK_ACCURACY:
            print(f" Model reached {EPOCH_BREAK_ACCURACY} accuracy, stopping training!")
            break

    return train_loss, train_acc


# Testing Function
def test_model(model, data_loader, device=None):
    """
    Tests a trained model on unseen data.

    Args:
        model: Trained PyTorch model
        data_loader: Testing dataset loader
        device: Device to run testing (CPU/GPU)

    Returns:
        accuracy: Accuracy of the model on the test dataset
    """
    if device is None:
        device = torch.device('cpu')

    model.eval()  # Set model to evaluation mode
    test_loss = 0
    correct = 0
    data_len = len(data_loader.dataset)

    with torch.no_grad():  # Disable gradient calculations for efficiency
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get most confident class
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions

    test_loss /= data_len
    accuracy = correct / data_len
    print(f"🧪 Test Set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{data_len} ({100 * accuracy:.2f}%)")

    return accuracy
