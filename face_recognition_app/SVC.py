from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# data = np.load("faces_embeddings.npz")
# embedded_x = data["arr_0"]
# y = data["arr_1"]
# print(y)

data = np.load("faces_embeddings.npz")
embedded_x, y = data["arr_0"], data["arr_1"]

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

x_train, x_test, y_train, y_test = train_test_split(embedded_x, y, shuffle=True)

model = SVC(kernel="linear", probability=True)
model.fit(x_train, y_train)

ypreds_train = model.predict(x_train)
ypreds_test = model.predict(x_test)

print(accuracy_score(y_train, ypreds_train))
print(accuracy_score(y_test, ypreds_test))

