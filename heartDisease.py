#import libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("D:/Amaya/SLIIT/Y3S2/FDM Project/FDM_Mini_Project_G38 - Copy1/FDM_Mini_Project_G38 - Copy1/Heart_Disease.csv")

x = data.drop('target',axis=1)

y = data['target']

# print(x,y)
#split set traing set 30% testingset 70%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
rand_forest = RandomForestClassifier()


#import pickel to train
rand_forest.fit(X_train, y_train)


#dump model in write mode-wb
pickle.dump(rand_forest,open('heartDiseaseModel.pkl','wb'))
#readable mode-rb
model=pickle.load(open('heartDiseaseModel.pkl','rb'))


