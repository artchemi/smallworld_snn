# smallworld_snn
## Описание проекта

## Установка
Все необходимые библиотеки указаны в файле requirements.txt, установка через pip.
```python
  pip install -r requirements.txt
```

## Запуск
Для запуска модели из статьи 
```python
  python train.py --learning_rate 0.2 --delta_tau 20
```
С ключом ```--hidden``` осторожнее, скорее всего там будут возникать ошибки.

Для запуска модели с малым миром
```python
  python train_sw.py --learning_rate 0.2 --delta_tau 20 --prob 0.5
```

## Результаты
Все результаты сохраняются в директорию ```data_d_m_y_time```. Параметры в названии папки - время запуска скриптов. Внутри создаются еще папки для первой и второй эпохи соответственно. ВАЖНО: В основной папке создаете ```.txt``` файл, в котором указано какие параметры использовались для обучения - скорость обучения, delta_tau, вероятность образования смежных ребер (для малого мира) и т.д. Там же после завершения скрипта будут метрики от RandomForest.

