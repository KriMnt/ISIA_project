import numpy as np
from sklearn import neural_network

#citim si importam fisierul cu baza de date modificata
data_test=np.genfromtxt(r'text.txt', delimiter=',')
data_train=np.genfromtxt(r'seg.txt', delimiter=',')



date_test=np.array(data_test[:,1:])
#selectam toate datele, mai putin prima coloana
etichete_test=np.array(data_test[:,0])
#selectam prima coloana, deoarece contine etichetele
date_train=np.array(data_train[:,1:])
#selectam toate datele, mai putin prima coloana
etichete_train=np.array(data_train[:,0])
#selectam prima coloana, deoarece contine etichetele

#CREARE SI ANTRENARE MLP
clf=neural_network.MLPClassifier(hidden_layer_sizes=(19,19),learning_rate_init=0.01,max_iter=1500)
clf.fit(date_train,etichete_train)

#TESTARE MLP
predictii=clf.predict(date_test)

acc=0
for i in range(len(etichete_test)):
    if etichete_test[i]==predictii[i]:
        acc=acc+1
print('Acuratetea=' + str((acc/len(etichete_test))*100) + '%')    


