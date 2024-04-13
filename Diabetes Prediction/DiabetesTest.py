from numpy import loadtxt
from keras.models import model_from_json


dataset = loadtxt(r'C:\Users\rithu\AppData\Local\Programs\Python\Python39\Programs\DiabetesPrediction\diabetes.csv' , delimiter=',' ,skiprows=1)
print(dataset)

x = dataset[:,0:8] ##Input->1 to 8 rows are inp
y = dataset[:,8] ##output -> after 8 is output

json_file = open('model.json' ,'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Loaded model from disk")

predictions = model.predict(x) #To predict 

#From coloumn 10 to 15 in dataset
for i in range(10,15):  #Prints input,pred op ,Expected op
    print('%s => %d (expected %d)' % (x[i].tolist(), predictions[i], y[i]))
