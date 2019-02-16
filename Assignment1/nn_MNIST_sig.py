import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F

root = './data'

if not os.path.exists(root):
    os.mkdir(root)


# Fully connected neural network with two hidden layers
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

    def name(self):
        return "MNIST_sig_h500"


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST Dataset
# Transforms - Normalizing data and converting it to Tensor form
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

# Training Data
training_data = torchvision.datasets.MNIST(root=root,
                                           train=True,
                                           transform=trans,
                                           download=True)
# Testing Data
testing_data = torchvision.datasets.MNIST(root=root,
                                          train=False,
                                          transform=trans,
                                          download=True)

# Data Loader
batch_size = 64

# Training data
training_loader = torch.utils.data.DataLoader(dataset=training_data,
                                              batch_size=batch_size,
                                              shuffle=True)
# Training data
testing_loader = torch.utils.data.DataLoader(dataset=testing_data,
                                             batch_size=batch_size)


lr = 0.00001

# Model
model = NeuralNet().to(device)

# Loss and optimizer
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

data_iter = iter(training_loader)
iter_per_epoch = len(training_loader)
total_step = 5000

loss_list = []
loss_dict = {}

# Start training
for step in range(total_step):

    # Reset the data_iter
    if (step + 1) % iter_per_epoch == 0:
        data_iter = iter(training_loader)

    # Grab images and labels
    images, labels = next(data_iter)
    images, labels = images.view(images.size(0), -1).to(device), labels.to(device)

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    loss_dict[step] = loss
    loss_list.append(loss.item())

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute accuracy
    _, argmax = torch.max(outputs, 1)
    accuracy = (labels == argmax.squeeze()).float().mean()

    if (step + 1) % 100 == 0:
        print(f'Step [{step+1}/{total_step}], Loss: {loss.item():.4}, Acc: {accuracy.item():.2}')

        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #

        # 1. Log scalar values (scalar summary)
        info = {'loss': loss.item(), 'accuracy': accuracy.item()}

        #for tag, value in info.items():
         #   writer.add_scalar(f'Train/{tag}', value, step + 1)

        # 2. Log values and gradients of the parameters (histogram summary)
        # for tag, value in model.named_parameters():
        #     tag = tag.replace('.', '/')
        #     logger.histo_summary(tag, value.data.cpu().numpy(), step + 1)
        #     logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step + 1)
        #
        # # 3. Log training images (image summary)
        info = {'images': images.view(-1, 28, 28)[:10].cpu().numpy()}
        #for tag, images in info.items():
            # logger.image_summary(tag, images, step + 1)
            #writer.add_images(tag, images, step + 1)

# Saving the model
torch.save(model.state_dict(), model.name())


# Test
model.eval()

test_iter = iter(testing_data)
images, labels = next(data_iter)

with torch.no_grad():
    #if args.cuda:
       # images, labels = images.cuda(), labels.cuda()

    outputs = model(images)
    print(outputs.shape)
    loss = criterion(outputs, labels)
    test_loss = loss.item()

# Compute accuracy
_, argmax = torch.max(outputs, 1)
accuracy = (labels == argmax.squeeze()).float().mean()

print(f'Loss: {test_loss:.4}, Acc: {accuracy:.2}')

