import os
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter

root = './data'

if not os.path.exists(root):
    os.mkdir(root)


# Fully connected neural network with two hidden layers
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

    def name(self):
        return "MNIST_relu_h500"




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
batch_size = 100

# Training data
training_loader = torch.utils.data.DataLoader(dataset=training_data,
                                              batch_size=batch_size,
                                              shuffle=True)
# Training data
testing_loader = torch.utils.data.DataLoader(dataset=testing_data,
                                             batch_size=batch_size)





# Model
model = NeuralNet().to(device)

# TensorboardX
writer = SummaryWriter('runs')

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                    help='learning rate (default: 0.00001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_dict[step] = loss
    loss_list.append(loss)

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

        for tag, value in info.items():
            writer.add_scalar(f'Train/{tag}', value, step + 1)

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
    if args.cuda:
        images, labels = images.cuda(), labels.cuda()

    outputs = model(images)
    print(outputs.shape)
    loss = criterion(outputs, labels)
    test_loss = loss.item()

# Compute accuracy
_, argmax = torch.max(outputs, 1)
accuracy = (labels == argmax.squeeze()).float().mean()

print(f'Loss: {test_loss:.4}, Acc: {accuracy:.2}')
