import matplotlib
from matplotlib import pyplot
from matplotlib import style
import numpy as np
import sklearn
from sklearn import datasets, linear_model

import pandas

# Load dataset
url = "speed_deneme.csv"
names = ['speed','value']
dataset = pandas.read_csv(url, names=names)

predict = 'value'

x = np.array(dataset.drop([predict], 1))
y = np.array(dataset[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size=0.2)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
	print( predictions[x], x_test[x], y_test[x]) 

	
p='speed'
style.use("ggplot")
pyplot.scatter(dataset[p],dataset['value'])
pyplot.xlabel(p)
pyplot.show()


#print(dataset.shape)
#print(dataset.head(20))

