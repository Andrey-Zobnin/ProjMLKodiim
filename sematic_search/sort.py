import pandas as pd
import json

# Загружаем Excel-файл
excel_file = 'input.xlsx'  
df = pd.read_excel(excel_file)

# Проверяем, какие там есть колонки
print("Найденные колонки:", df.columns)

data_list = []
for _, row in df.iterrows():
    item = {
        'question': str(row['Вопрос']).strip(),
        'answer': str(row['Ответ']).strip(),
        'category': str(row['Категория']).strip()
    }
    data_list.append(item)

# Формируем итоговую структуру
output_data = {'data': data_list}

# Сохраняем в JSON
with open('qa_data.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("Готово! Данные сохранены в qa_data.json")
