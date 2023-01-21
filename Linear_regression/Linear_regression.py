import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn  import datasets , linear_model
from sklearn.metrics import mean_squared_error
import pickle

diabates = datasets.load_diabetes()
cols =['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'] 

diabates_x = diabates.data#[: ,np.newaxis,2]

diabates_x_train =diabates_x[:-30]
diabates_x_test = diabates_x[-20:]

diabates_y_train =diabates_x[:-30]
diabates_y_test = diabates_x[-20:]

model = linear_model.LinearRegression() 
model.fit(diabates_x_train,diabates_y_train)

with open("linearmodel.pickle","wb") as f:
    pickle.dump(model,f)                            #Saving the daata

# pickle_in =  open("linearmodel.pickle","rb")
# model = pickle.load(pickle_in)                    #Loading the data


diabates_y_predict = model.predict(diabates_x_test)

print("Mean squared error :" , mean_squared_error(diabates_y_test,diabates_y_predict))

print("weights :",model.coef_ ,"\n","Intercept :",model.intercept_)

plt.scatter(diabates_x_test,diabates_y_test)
plt.plot(diabates_x_test,diabates_y_predict)
plt.show()

# nts/Machine learning/Linear_regression.py"
# Mean squared error : 1.8178056230099e-33
# weights : [[1.]]
#  Intercept : [-9.48676901e-20]

#  1.3945098721799241e-33