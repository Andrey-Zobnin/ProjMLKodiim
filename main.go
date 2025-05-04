package main

import (
	"crypto/tls"
	"fmt"
	"log"
	"log/slog"
	"net/http"
	"os"

	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/jmoiron/sqlx"
	"project/api"
	"project/storage"
)

func main() {
	logger := slog.Default()

	pgURL := os.Getenv("POSTGRES_CONN")
	if pgURL == "" {
		pgURL = "postgres://user:password@localhost:5432/mydb?sslmode=disable"
		logger.Warn("using hardcoded POSTGRES_CONN for development")
	}

	db, err := sqlx.Connect("pgx", pgURL)
	if err != nil {
		log.Fatalln(err)
	}
	defer db.Close()
	storage.SetPostgres(db)

	// 🛡️ JWT секрет
	secret := os.Getenv("RANDOM_SECRET")
	if secret == "" {
		secret = "mydevelopmentsecret"
		logger.Warn("using hardcoded RANDOM_SECRET for development")
	}
	api.JWTSecretKey = []byte(secret)

	// 🌐 TLS-сервер
	serverAddress := os.Getenv("SERVER_ADDRESS")
	if serverAddress == "" {
		serverAddress = "localhost:8443"
		logger.Warn("using hardcoded SERVER_ADDRESS for development")
	}

	s := api.NewServer(serverAddress, logger)

	// TLS ключи
	certFile := "cert/server.crt"
	keyFile := "cert/server.key"

	tlsConfig := &tls.Config{
		MinVersion: tls.VersionTLS13,
	}

	server := &http.Server{
		Addr:      serverAddress,
		Handler:   s.Router,
		TLSConfig: tlsConfig,
	}

	fmt.Println("🔐 TLS-сервер слушает на https://" + serverAddress)
	err = server.ListenAndServeTLS(certFile, keyFile)
	if err != nil {
		logger.Error("TLS сервер остановлен", "error", err)
	}
}