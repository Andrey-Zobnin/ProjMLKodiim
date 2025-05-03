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

	pgURL := os.Getenv("POSTGRES_CONN")
	if pgURL == "" {
		logger.Error("missed POSTGRES_CONN env")
		os.Exit(1)
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
		logger.Error("missed SERVER_ADDRESS env var")
		os.Exit(1)
	}

	api.JWTSecretKey = []byte(os.Getenv("RANDOM_SECRET"))

	storage.SetPostgres(db)

	s := api.NewServer(serverAddress, logger)

	err = s.Start()
	if err != nil {
		logger.Error("server has been stopped", "error", err)
	}
}
