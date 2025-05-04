package socnetwork

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// MLService предоставляет методы для взаимодействия с внешним ML API
type MLService struct {
	PythonEndpoint string
}

// NewMLService создает новый экземпляр MLService
func NewMLService(endpoint string) *MLService {
	return &MLService{PythonEndpoint: endpoint}
}

// SearchRequest представляет запрос к ML API
type SearchRequest struct {
	TopK  int    `json:"top_k"`
	Query string `json:"query"`
}

// SearchResult представляет ответ от ML API
type SearchResult struct {
	Results []string `json:"results"`
}

// SemanticSearch отправляет запрос к ML-сервису и возвращает релевантные результаты
func (ml *MLService) SemanticSearch(ctx context.Context, query string, topK int) ([]string, error) {
	reqBody := SearchRequest{
		Query: query,
		TopK:  topK,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := http.Post(ml.PythonEndpoint+"/search", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to send request to ML service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ML service returned status %d: %s", resp.StatusCode, string(data))
	}

	var result SearchResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result.Results, nil
}

func CallML(mlURL, text string) (string, error) {
	payload := map[string]string{"text": text}
	body, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("не смог сериализовать запрос: %w", err)
	}

	resp, err := http.Post(mlURL, "application/json", bytes.NewBuffer(body))
	if err != nil {
		return "", fmt.Errorf("ошибка запроса к ML: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("ml вернул ошибку: %s", resp.Status)
	}

	var response struct {
		Result string `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return "", fmt.Errorf("не смог декодировать ответ ML: %w", err)
	}

	return response.Result, nil
}
