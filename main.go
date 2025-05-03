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

// Config структура для хранения конфигурации приложения
type Config struct {
	PostgresURL   string
	ServerAddress string
	JWTSecretKey  []byte
}

func loadConfig(logger *slog.Logger) Config {
	pgURL := os.Getenv("POSTGRES_CONN")
	if pgURL == "" {
		pgURL = "postgres://user:password@localhost:5432/mydb?sslmode=disable"
		logger.Warn("using hardcoded POSTGRES_CONN for development")
	}

	serverAddress := os.Getenv("SERVER_ADDRESS")
	if serverAddress == "" {
		serverAddress = "localhost:8080"
		logger.Warn("using hardcoded SERVER_ADDRESS for development")
	}

	secret := os.Getenv("RANDOM_SECRET")
	if secret == "" {
		secret = "mydevelopmentsecret"
		logger.Warn("using hardcoded RANDOM_SECRET for development")
	}

	return Config{
		PostgresURL:   pgURL,
		ServerAddress: serverAddress,
		JWTSecretKey:  []byte(secret),
	}
}

func main() {
	logger := slog.Default()

	// Загружаем конфигурацию
	config := loadConfig(logger)

	db, err := sqlx.Connect("pgx", config.PostgresURL)
	if err != nil {
		log.Fatalln(err)
	}
	defer func() {
		_ = db.Close()
	}()

	// TODO добавить проверку соединения с БД, например db.Ping()
	storage.SetPostgres(db)

	api.JWTSecretKey = config.JWTSecretKey

	s := api.NewServer(config.ServerAddress, logger)

	// TODO Можно добавить graceful shutdown через signal.Notify
	err = s.Start()
	if err != nil {
		logger.Error("server has been stopped", "error", err)
	}
}
