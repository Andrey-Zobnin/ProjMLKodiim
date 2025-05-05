package api

import (
    "encoding/json"
    "io"
    "net/http"
    "project/internal/mlclient"
    "log/slog"
)

func MLHandler(w http.ResponseWriter, r *http.Request) {
    var payload map[string]interface{}
    err := json.NewDecoder(r.Body).Decode(&payload)
    bodyBytes, err := json.Marshal(payload)
    resp, err := mlclient.SendToML(bodyBytes)
    
    // Check if the response is valid
    if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
        slog.Error("Invalid JSON payload", "error", err)
        
        http.Error(w, "Invalid JSON payload", http.StatusBadRequest)
        return
    }
    slog.Info("Received payload", "payload", payload)
    if err != nil {
        slog.Error("Failed to marshal payload", "error", err)
        http.Error(w, "Failed to process payload", http.StatusInternalServerError)
        return
    }
    slog.Info("Sending payload to ML", "payload", payload)
    // Send the payload to the ML service
    if err != nil {
        slog.Error("Failed to read ML response body", "error", err)
        http.Error(w, "Failed to read ML response", http.StatusInternalServerError)
        return
    }
    
    slog.Info("Received response from ML", "status", resp.StatusCode, "response", string(responseBody))
    defer resp.Body.Close() // close the response body

    // Отдаём клиенту
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(resp.StatusCode)
    io.Copy(w, bytes.NewReader(responseBody))

}
