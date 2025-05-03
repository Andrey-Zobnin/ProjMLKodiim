package api

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
)

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		panic(fmt.Errorf("json ошибка: %w", err))
	}
}

func InternalError(w http.ResponseWriter, l *slog.Logger, err error) {
	if l != nil {
		l.Error(err.Error())
	}
	LogicError(w, http.StatusInternalServerError, err.Error())
}

func BadRequest(w http.ResponseWriter, reason string) {
	LogicError(w, http.StatusBadRequest, reason)
}

func LogicError(w http.ResponseWriter, code int, reason string) {
	// struct to response
	type Response struct {
		Reason string `json:"reason"`
	}
	resp := Response{Reason: reason}
	writeJSON(w, code, &resp)
}
