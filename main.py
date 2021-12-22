import numpy as np
from sklearn.datasets import load_iris


class Data:

    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __del__(self):
        Data.__instance = None  # удаляем единственный объект класса Data

    def __init__(self, data_stamp: tuple):
        """data - кортеж из нумпаевских многомерных массивов X и y"""
        self.X, self.y = data_stamp

    def transform(self):
        means = self.X.mean(axis=0)
        stds = self.X.std(axis=0)
        return (self.X - means) / stds

    def train_test_split(self, train_size=0.7, random_state=42) -> tuple:
        np.random.seed(random_state)
        size = int(train_size*self.X.shape[0])
        mask = np.random.choice(range(self.X.shape[0]), size=size)
        return self.X[mask], self.y[mask]


data = load_iris(return_X_y=True)
d1 = Data(data)
d2 = Data((data[0]+1, data[1]+1))
print(d1 is d2)
# print(id(d1), id(d2))
# print(d1.transform())

