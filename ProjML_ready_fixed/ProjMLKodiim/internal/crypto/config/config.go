package config

import "os"

type Config struct {
	PostgresURL   string
	JWTSecret     string
	ServerAddress string
}

func Load() *Config {
	return &Config{
		// todo check local host @ssd2008
		PostgresURL:   getEnv("POSTGRES_CONN", "postgres://user:pass@localhost:5432/db?sslmode=disable"),
		JWTSecret:     getEnv("RANDOM_SECRET", "devsecret"),
		ServerAddress: getEnv("SERVER_ADDRESS", "localhost:8443"),
	}
}

func getEnv(key string, fallback string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return fallback
}
