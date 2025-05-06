
package api

import (
    "encoding/json"
    "fmt"
    "net/http"
    "strings"

    "github.com/golang-jwt/jwt/v4"
)

var jwtKey = []byte("supersecretkey")

type Claims struct {
    Username string `json:"username"`
    jwt.RegisteredClaims
}

type MLRequest struct {
    Input string `json:"input"`
}

type MLResponse struct {
    Prediction string `json:"prediction"`
}

func AuthMiddleware(next http.HandlerFunc) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        tokenStr := r.Header.Get("Authorization")
        if !strings.HasPrefix(tokenStr, "Bearer ") {
            http.Error(w, "Unauthorized", http.StatusUnauthorized)
            return
        }

        tokenStr = strings.TrimPrefix(tokenStr, "Bearer ")
        claims := &Claims{}
        token, err := jwt.ParseWithClaims(tokenStr, claims, func(token *jwt.Token) (interface{}, error) {
            return jwtKey, nil
        })
        if err != nil || !token.Valid {
            http.Error(w, "Invalid token", http.StatusUnauthorized)
            return
        }

        next(w, r)
    }
}

func MLHandler(w http.ResponseWriter, r *http.Request) {
    var req MLRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "Invalid input", http.StatusBadRequest)
        return
    }

    // Заглушка для запроса к ML-сервису
    prediction := "Processed: " + req.Input
    res := MLResponse{Prediction: prediction}

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(res)
}
