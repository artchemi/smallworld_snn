# smallworld_snn
## Описание

## Установка
Все необходимые библиотеки указаны в файле requirements.txt, установка через pip.
```python
  pip install -r requirements.txt
```

## Запуск
Моделирование можно провести с помощью ```script_ws.py```, указывая во флагах необходимые параметры. 

n - число нейронов 

k - число соседей для каждого нейрона (k < n)

p - вероятность перестройки каждого ребра

generator_type - тип генератора, см. по примерам

steps - число шагов

dt - шаг интегрирования (лучше всего 0.1 или 0.01)
