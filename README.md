# Мастер ударений

## Мотивация
Повышение орфоэпической грамотности в интерактивной форме

## План с датами
- 14.01 - подготовка словаря и презентации
- 15.01 - презентация выбранной темы
- 16.01 - запись данных + обучение модели
- 17.01 - 18.01 - обучение модели, получение бейзлайна
- 20.01 - 24.01 - разработка сайта (фронт, бэк), дообучение модели
- 25.01 - защита проекта

## Методы
- *Whisper* - предварительно обученная модель для распознавания речи
- *Fine-tuning*
- *Multi-stage training*

## Данные
https://colab.research.google.com/drive/1Mly0VOb2_JryqBZqxRlGnjjeS8xVqm9h?usp=sharing
Для бейзлайна записываем самостоятельно, а для готового продукта - из внешних источников:
- Корпусы аудио- и текстовых данных
- Специализированные орфоэпические словари
- Аудиокниги
- Платформы с пользовательскими записями речи
- Готовые датасеты для обучения моделей

## Метрики
*Частота ошибок в словах (Word Error Rate, WER)*:
Измеряет долю ошибок на уровне слов, включая замены, вставки и удаления. Это означает, что ошибки аннотируются на уровне каждого слова 
Формула: *WER = (S + I + D) / N*
- S — количество замен,
- I — количество вставок,
- D — количество удалений,
- N — общее количество слов в эталонной последовательности.
Низкий WER указывает на высокую точность модели.

*Скорость обработки (Inference Speed)*
Оценка времени, необходимого модели для обработки аудиофайла.
Измеряется в реальном времени, например, секунд обработки на секунду аудио (RTF — Real-Time Factor).

