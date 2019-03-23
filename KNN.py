import matplotlib
from matplotlib import pyplot
from matplotlib import style
import numpy as np
import pandas
import sklearn
from sklearn import datasets, linear_model, preprocessing, neighbors


url = "file3.csv"
names = ['e_load', 'RPM', 'speed','value']
dataset = pandas.read_csv(url, names=names)

#print(dataset)

le = preprocessing.LabelEncoder()
e_load = le.fit_transform(list(dataset['e_load']))
RPM = le.fit_transform(list(dataset['RPM']))
speed = le.fit_transform(list(dataset['speed']))
classes = le.fit_transform(list(dataset['value']))

predict = 'value'

x = list(zip(e_load, RPM, speed))
y = list(classes)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.2)

model = neighbors.KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print ("Model score: ", acc)

predicted = model.predict(x_test)
names =["bad", "normal"]

for x in range(len(predicted)):
	print( "Data: ", x_test[x], "\t predicted: ", names[predicted[x]-1],  "\t Actual: ", names[y_test[x]-1] )  


# p='speed'
# style.use("ggplot")
# pyplot.scatter(dataset[p],dataset['value'])
# pyplot.xlabel(p)
# pyplot.show()


