from numpy import loadtxt ##loading the excel file
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

dataset = loadtxt(r'C:\Users\rithu\AppData\Local\Programs\Python\Python39\Programs\DiabetesPrediction\diabetes.csv', delimiter=',', skiprows=1)

#print(dataset)

x = dataset[:,0:8] ##Input->1 to 8 rows are inp
y = dataset[:,8] ##output -> after 8 is output

##print("Input",x)
##print("Output",y)

model = Sequential()

model.add(Dense(12, input_dim=8,activation ='relu'))##8 bacause input rows are 8
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation ='sigmoid')) ##1 because op is binary,So sigmoid is used

model.compile(loss='binary_crossentropy' , optimizer ='adam', metrics=['accuracy'])

#model training

model.fit(x,y, epochs =40 , batch_size=10)

#Evaluation

_,accuracy = model.evaluate(x,y)
print('Accuracy :%2f' % (accuracy*100))

#Model save
model_json = model.to_json()
with open("model.json" ,"w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")


