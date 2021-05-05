import tensorflow as tf
import numpy as np

a = (1,2,3,)
print(*a)

# 模拟数据
def GenerateData(data_size=100):
    train_x = np.linspace(-1, 1, data_size)
    print(*train_x.shape)
    train_y = 2 * train_x + np.random.randn(*train_x.shape) * 0.2
    return train_x,train_y

# 生成模拟数据
train_data = GenerateData()

print(train_data)

# 获取一个数据
def get_one(dataset):
    for elment in dataset:
        return elment

# 显示一个数据
def show_elment(elment):
    x, y = elment
    print("x shape:", x.shape)
    print("x:", x.numpy())
    print("y shape:", y.shape)
    print("y:", y.numpy())

# 显示头5个数据
def show_head(dataset, size=5):
    for step, elment in dataset.enumerate():
        show_elment(elment)
        if step >= size-1:
            break

# 以元组方式 生成Dataset数据集
batch_size = 10
dataset_tuple = tf.data.Dataset.from_tensor_slices(train_data)
db_tuple = dataset_tuple.shuffle(100).batch(batch_size)

# 显示元组数据集
elment_tuple = get_one(db_tuple)
# show_elment(elment_tuple)

show_head(db_tuple, 1)
