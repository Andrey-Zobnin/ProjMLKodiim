
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
    if err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }

    bodyBytes, err := json.Marshal(payload)
    if err != nil {
        http.Error(w, "Failed to marshal JSON", http.StatusInternalServerError)
        return
    }

    resp, err := mlclient.SendToML(bodyBytes)
    if err != nil {
        slog.Error("ML request failed", "error", err)
        http.Error(w, "ML service error", http.StatusBadGateway)
        return
    }
    defer resp.Body.Close()

    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(resp.StatusCode)
    io.Copy(w, resp.Body)
}
