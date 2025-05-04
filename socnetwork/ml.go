package socnetwork

import (
	"fmt"
	"bytes"
	"context"
	"net/http"
	"encoding/json"
	"io"
)

type MlService struct {
	PuthonEndpoint string
}

func Mlservice(endpoint string) *Mlservice {
	return &MlService{PythonEndpoint: endpoint}
}

type SearchRequest struct {
	TopK  int    `json:"top_k"`
	Query string `json:"query"`
}

type SearchResult struct {
	Results []string `json:"results"`
}

func (ml *MLService) SemanticSearch(ctx context.Context, query string, topK int) ([]string, error) {
	reqBody := SearchRequest{
		Query: query,
		TopK:  topK,
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	resp, err := http.Post(ml.PythonEndpoint+"/search", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		data, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Python API error: %s", data)
	}

	var result SearchResult
	err = json.NewDecoder(resp.Body).Decode(&result)
	return result.Results, err
}