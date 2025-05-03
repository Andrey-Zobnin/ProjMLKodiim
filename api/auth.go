package api

import (
	"github.com/golang-jwt/jwt/v5"
)

// TODO: Реализовать механизм отзыва токенов
var JWTSecretKey = []byte(``)

type Credentials struct {
	Login    string `json:"login"`
	Password string `json:"password"`
}

// TODO: Добавить валидацию Claims (issuer, audience)
// Claims содержит данные закодированные в токен.
// jwt.RegisteredClaims встроен так как там есть время экспирации
type Claims struct {
	Login string `json:"username"`
	Tag   string `json:"tag"`
	// тут хеш пароля что обеспечит деактивацию токена при смене пароля
	// что такое jwt - нет в импортах
	jwt.RegisteredClaims
	// TODO: Использовать sync.Once для инициализации JWTSecretKey
}
