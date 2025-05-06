# Создание виртуального окружения

## 1. Создание виртуального окружения

Откройте терминал в директории с вашим проектом и выполните:

bash
python -m venv venv

## 2. Активация виртуального окружения

- На **Windows**:

  
bash
  venv\Scripts\activate
  

- На **macOS/Linux**:

  
bash
  source venv/bin/activate
  

## 3. Установка библиотек

Если у вас уже есть список библиотек, например, в requirements.txt, выполните:

bash
pip install -r requirements.txt

Если списка нет, и вы знаете названия библиотек, например:


flask
requests
pandas

то выполните:

bash
pip install flask requests pandas

(Замените названия на свои.)

## 4. Запуск app.py

После установки библиотек, запустите файл:

bash
python app.py
