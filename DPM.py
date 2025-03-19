#import section:
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense

#data allocation:
data = np.loadtxt("pima-indians-diabetes.csv",delimiter=',')
x_train = data[:760,:8]
y_train  = data[:760,-1]
x_test = data[760:,:8]
y_test = data[760:,-1]

# model bulding:
model = Sequential()
model.add(Dense(100,activation='relu',input_shape=(8,)))
model.add(Dense(200,activation='relu'))
model.add(Dense(300,activation='relu'))
model.add(Dense(150,activation='relu'))
model.add(Dense(250,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer="adam",metrics=["accuracy"])
model.fit(x_train,y_train,epochs=350,batch_size=10)
model.save("model_diabetes.txt")

# After saving the model comment the model and run model test
# trained model will be saved model_diabetes.txt
# you can also save as model.save("model_diabetes.h5")   Corrected file format
# Keras models should be saved in HDF5 (.h5) or TensorFlow SavedModel format.

# model test:
# model= load_model("model_diabetes.txt")
# prediction = model.predict(x_test)

# model result:
# l=[]
# for i in prediction:
#     if i>= 0.5:
#         l.append(1)
#     else:
#         l.append(0)
# print(np.array(l))

# print(prediction)
# print(y_test.astype("int32"))