# -*- coding: utf-8 -*-
import pickle
# Импортируем все пакеты, которые необходимы для вашей модели
import numpy as np
import pandas as pd
from sklearn import linear_model
import os
# Импортируем Flask для создания API
from flask import Flask, request

# Загружаем модель в память
with open('./model.pkl', 'rb') as model_pkl:
   knn = pickle.load(model_pkl)

# Инициализируем приложение Flask
app = Flask(__name__)

# Создайте конечную точку API
@app.route('/predict')
def predict_iris():
   # Считываем все необходимые параметры запроса
   sl = request.args.get('sl')
   sw = request.args.get('sw')
   pl = request.args.get('pl')
   pw = request.args.get('pw')

# Используем метод модели predict для
# получения прогноза для неизвестных данных
   unseen = pd.DataFrame({"sepal length (cm)" : [sl],
                   "sepal width (cm)" :[sw],
                   "petal length (cm)":[pl],
                   "petal width (cm)":[pw]})
   result = knn.predict(unseen)
  # возвращаем результат
   return 'Predicted result for observation ' + str(unseen) + ' is: ' + str(result)
if __name__ == '__main__':
   app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

