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
    
    if err != nil {
        http.Error(w, "Invalid JSON or error datd", http.StatusBadRequest)
        return
    } else if err != nil {
        http.Error(w, "Failed to marshal JSON", http.StatusInternalServerError)
        return
    } else {
        slog.Error("ML request failed", "error", err)
        http.Error(w, "ML service error", http.StatusBadGateway)
        return
    }
    // defualt case
    defer resp.Body.Close() // close the response body

    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(resp.StatusCode)
    io.Copy(w, resp.Body)
}
