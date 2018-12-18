#################
#Date: 2018-12-17
#Authur: Eva
#Topic: CNN
#################


# standard library
import os
# 3-party library
import torch
import torch.nn as nn
import torch.utilis.data as Data
import torchvision as vision
import matplotlib.pyplot as plt


# torch.manual_seed(1)

# Hyperparameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

# MNIST dataset
if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
	DOWNLOAD_MNIST = True

train_data = vision.dataset.MNIST(
	root = './mnist/',
	train = True,
	transform = vision.transform.ToTensor(),
	download = DOWNLOAD_MNIST,
)

# plot one example
print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[0].numpy(), cmap='grey')
plt.title('%i' % train_data.train_labels[0])
plt.show()

# Data loader for easy mini-batch return in training, the image batch shape will be (50,1,28,28)
train_loader = Data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True)

# Pick 2000 samples
test_data = vision.dataset.MNIST(root='./MNIST/', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/225.
test_y = test_data.test_labels[:2000]

# Network Architecture
class CNN(nn.module):
	"""docstring for CNN"""
	def __init__(self):
		super(CNN, self).__init__()

		self.conv1 = nn.Sequential(
			nn.conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2,),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),
		)

		self.conv2 = nn.Sequential(
			nn.conv2d(16,32,5,1,2),
			nn.ReLU(),
			nn.MaxPool2d(2),
		)

		self.out = nn.Linear(32*7*7, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)
		output = self.out(x)
		return output, x

cnn = CNN()
print(cnn)

# Optimizer
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# Training
for epoch in range(EPOCH):
	for step, (b_x, b_y) in enumerate(train_loader):
		output = cnn(b_x)[0]
		loss = loss_func(output, b_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	if step % 50 == 0:
		test_output, last_layer = cnn(test_x)
		pred_y = torch.max(test_output, 1)[1].data_numpy()
		accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum())/float(test_y.size(0))
		print('Epoch: ', epoch, '| Train_loss: %.4f' % loss.data_numpy(), '| Test_accuracy: %.2f' % accuracy)

test_output, _ = cnn(test_x[;10])
pred_y = torch.max(test_output,1)[1]data_numpy()
print(pred_y, 'prediction number: ')
print(test_y[:10].numpy(), ' real number')













