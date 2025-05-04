# Документация пакета socnetwork

## Описание

Пакет socnetwork предоставляет интерфейс для взаимодействия с внешним ML API, который выполняет семантический поиск. Он включает в себя структуру MLService, которая содержит методы для отправки запросов и получения результатов от ML-сервиса.

## Структура пакета


/socnetwork
├── ml.go 
├── socetnetwork.md # маркдаун документации файла ml.go      
└── user.go # файл пользователя

## Установка

Для использования пакета socnetwork в вашем проекте, добавьте его в зависимости вашего проекта:

bash
go get github.com/ваш-репозиторий/socnetwork

## Использование

### Создание нового экземпляра MLService

Чтобы создать новый экземпляр MLService, используйте функцию NewMLService, передав URL-адрес вашего ML API:

go
mlService := NewMLService("http://localhost:5000")

### Метод SemanticSearch

#### Описание

Метод SemanticSearch отправляет запрос к ML-сервису и возвращает релевантные результаты на основе заданного запроса.

#### Сигнатура

go
func (ml *MLService) SemanticSearch(ctx context.Context, query string, topK int) ([]string, error)

#### Параметры

- ctx: контекст для управления временем выполнения запроса.
- query: строка, представляющая запрос для семантического поиска.
- topK: количество наиболее релевантных результатов, которые необходимо вернуть.

#### Возвращаемые значения

- []string: массив строк, представляющий результаты поиска.
- error: ошибка, если запрос не удался или если произошла ошибка при обработке ответа.

#### Пример использования

go
ctx := context.Background()
results, err := mlService.SemanticSearch(ctx, "поиск по тексту", 5)
if err != nil {
    log.Fatalf("Ошибка при выполнении поиска: %v", err)
}
fmt.Println("Результаты поиска:", results)

## Структуры данных

### SearchRequest

Структура SearchRequest представляет собой запрос к ML API.

go
type SearchRequest struct {
    TopK  int    json:"top_k"
    Query string json:"query"
}

#### Поля

- TopK: количество результатов, которые нужно вернуть.
- Query: текст запроса для поиска.

### SearchResult

Структура SearchResult представляет собой ответ от ML API.

go
type SearchResult struct {
    Results []string json:"results"
}

#### Поля

- Results: массив строк, содержащий результаты поиска.

## Заключение