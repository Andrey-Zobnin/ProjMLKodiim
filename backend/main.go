
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "bytes"
)

type Message struct {
    Content string `json:"content"`
}

func handler(w http.ResponseWriter, r *http.Request) {
    var msg Message
    err := json.NewDecoder(r.Body).Decode(&msg)
    if err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    mlResp, err := http.Post("http://ml_service:5000/process", "application/json", bytes.NewBuffer([]byte(`{"content":"`+msg.Content+`"}`)))
    if err != nil {
        http.Error(w, "ML service failed", http.StatusInternalServerError)
        return
    }
    defer mlResp.Body.Close()

    var mlMsg Message
    json.NewDecoder(mlResp.Body).Decode(&mlMsg)

    json.NewEncoder(w).Encode(mlMsg)
}

func main() {
    http.HandleFunc("/chat", handler)
    fmt.Println("Backend running on port 8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
