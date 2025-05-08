package main

import (
	"log"
	"log/slog"
	"os"

	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/jmoiron/sqlx"

	"project/api"
	"project/storage"
)

func main() {
	logger := slog.Default()

	// временный хардкод для разработки
	pgURL := os.Getenv("POSTGRES_CONN")
	if pgURL == "" {
		pgURL = "postgres://user:password@localhost:5432/mydb?sslmode=disable"
		logger.Warn("using hardcoded POSTGRES_CONN for development")
	}

	db, err := sqlx.Connect("pgx", pgURL)
	if err != nil {
		log.Fatalln(err)
	}
	defer func() {
		_ = db.Close()
	}()

	serverAddress := os.Getenv("SERVER_ADDRESS")
	if serverAddress == "" {
		serverAddress = "localhost:8080"
		logger.Warn("using hardcoded SERVER_ADDRESS for development")
	}

	var mlURL = os.Getenv("ML_URL")
	if mlURL == "" {
		mlURL = "localhost:9000"
		logger.Warn("using hardcoded ML_URL for development")
	}

	secret := os.Getenv("RANDOM_SECRET")
	if secret == "" {
		secret = "mydevelopmentsecret"
		logger.Warn("using hardcoded RANDOM_SECRET for development")
	}
	api.JWTSecretKey = []byte(secret)

	storage.SetPostgres(db)

	s := api.NewServer(serverAddress, logger)

	err = s.Start()
	if err != nil {
		logger.Error("server has been stopped", "error", err)
	}
}
