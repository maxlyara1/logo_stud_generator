# Как запустить веб-интерфейс на локальном устройстве?
### Ввести следующие команды в консоль (при первом запуске все, далее - при повторных запусках - выполнить только 4 шаг, соответственно модель сможет работать автономно т.к. все библиотеки ранее уже были загружены):
1. python -m venv venv
2. * venv\Scripts\activate - для Windows
   * source venv/bin/activate - для Макбука
3. pip install -r requirements.txt
4. streamlit run app.py
