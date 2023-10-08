import numpy as npcls
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error ,mean_absolute_error, r2_score
import torch
from d2l import torch as d2l
from torch import nn
import numpy as np
# 读取标准化后的数据
train_data = pd.read_csv('./clean_data.csv')
train_data = train_data.drop(["constructionTime","Lng","Lat"], axis=1)
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
all_features = train_data
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values,
							  dtype=torch.float32)
train_labels = torch.tensor(train_data.totalPrice.values.reshape(-1, 1),
							dtype=torch.float32)
import matplotlib.pyplot as plt

# 从CSV文件中读取数据
# data = pd.read_csv('./111.csv')

# # 提取需要用于散点图的列数据
# x = data['month']  # 用实际的列名替换'ColumnName_X'
# y = data['totalPrice']  # 用实际的列名替换'ColumnName_Y'

# # 绘制散点图
# plt.scatter(x, y)

# # 添加标题和轴标签
# plt.title('Scatter Plot Example')
# plt.xlabel('X-axis Label')
# plt.ylabel('Y-axis Label')

# # 显示图形
# plt.show()

# input()
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
	net = nn.Sequential(nn.Linear(in_features, 1))
	return net

def log_rmse(net, features, labels):
	clipped_preds = torch.clamp(net(features), 1, float('inf'))
	rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
	return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
		  num_epochs, learning_rate, weight_decay, batch_size):
	train_ls, test_ls = [], []
	train_losses = []
	train_iter = d2l.load_array((train_features, train_labels), batch_size)
	optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
								 weight_decay=weight_decay)
	for epoch in range(num_epochs):
		for X, y in train_iter:
			optimizer.zero_grad()
			l = loss(net(X), y)
			l.backward()
			optimizer.step()
		train_loss = log_rmse(net, train_features, train_labels)
		train_losses.append(train_loss)
		train_ls.append(log_rmse(net, train_features, train_labels))
		if test_labels is not None:
			test_ls.append(log_rmse(net, test_features, test_labels))
			
	plt.figure(figsize=(10, 6))
	plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
	if test_labels is not None:
		plt.plot(range(1, num_epochs + 1), test_ls, label='Test Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Log RMSE Loss')
	plt.legend()
	plt.grid()
	plt.show()
	return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
	assert k > 1
	fold_size = X.shape[0] // k
	X_train, y_train = None, None
	for j in range(k):
		idx = slice(j * fold_size, (j + 1) * fold_size)
		X_part, y_part = X[idx, :], y[idx]
		if j == i:
			X_valid, y_valid = X_part, y_part
		elif X_train is None:
			X_train, y_train = X_part, y_part
		else:
			X_train = torch.cat([X_train, X_part], 0)
			y_train = torch.cat([y_train, y_part], 0)
	return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
		   batch_size):
	train_l_sum, valid_l_sum = 0, 0
	for i in range(k):
		data = get_k_fold_data(k, i, X_train, y_train)
		net = get_net()
		train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
								   weight_decay, batch_size)
		train_l_sum += train_ls[-1]
		valid_l_sum += valid_ls[-1]
		if i == 0:
			d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
					 xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
					 legend=['train', 'valid'], yscale='log')
		print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
			  f'vali d log rmse {float(valid_ls[-1]):f}')
	return train_l_sum / k, valid_l_sum / k

k, num_epochs, lr, weight_decay, batch_size = 5, 1000, 0.001, 0, 5000
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
						  weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
	  f'平均验证log rmse: {float(valid_l):f}')

def train_and_pred(train_features, train_labels,
				   num_epochs, lr, weight_decay, batch_size):
	net = get_net()
	train_ls, _ = train(net, train_features, train_labels, None, None,
						num_epochs, lr, weight_decay, batch_size)
	d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
			 ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
	print(f'train log rmse {float(train_ls[-1]):f}')
	

train_and_pred(train_features, train_labels,
			   num_epochs, lr, weight_decay, batch_size)

params = list(get_net().parameters())

# 线性层的权重参数
linear_layer_weights = params[0].data
print("Linear Layer Weights:")
print(linear_layer_weights)

# 线性层的偏置参数
linear_layer_bias = params[1].data
print("Linear Layer Bias:")
print(linear_layer_bias)