import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = \
	load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []
train_size = x_train.shape[0]

#ハイパーパラメータ
inters_num = 10000
batch_size = 100
learning_late = 1

# エポックあたりの繰り返し数
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

x = []
j = -1
plt.title("learning_late = 1")
plt.xlabel("trial")
plt.ylabel("accuracy")
for i in range(inters_num):
	# ミニバッチの取得
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]

	# 勾配の計算
	grad = network.gradient(x_batch, t_batch)

	# パラメータの更新
	for key in ('W1', 'b1', 'W2', 'b2'):
		network.params[key] -= learning_late * grad[key]

	# 学習経過の記録
	loss = network.loss(x_batch, t_batch)
	train_loss_list.append(loss)



	# 1エポックごとに認識精度を計算
	if i % iter_per_epoch == 0:
		train_acc = network.accuracy(x_train, t_train)
		test_acc = network.accuracy(x_test, t_test)
		train_acc_list.append(train_acc)
		test_acc_list.append(test_acc)
		print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
		j += 1
		x.append(j)
		if j == 0:
			axis1,  = plt.plot(x, train_acc_list, label = "train")
			axis2,  = plt.plot(x, test_acc_list, label = "test")
			plt.xlim(0, 1)
			plt.ylim(0, 1)
			plt.legend()
			plt.ion()
			plt.draw()
			plt.pause(0.01)
		# 折れ線グラフを再描画する
		if j > 0:
			axis1.set_data(x, train_acc_list)
			axis2.set_data(x, test_acc_list)
			plt.xlim(0, j)
			plt.draw()
			plt.pause(0.01)
		if j == 16:
			plt.pause(20)
