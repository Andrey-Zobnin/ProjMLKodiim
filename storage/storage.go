package storage

import (
	"github.com/jmoiron/sqlx"
)

var postgres *sqlx.DB

func SetPostgres(db *sqlx.DB) {
	postgres = db
}

func GetPostgres() *sqlx.DB {
	if postgres == nil {
		panic("вызовите SetPostgres")
	}
	return postgres
}

var userStorage *UserStorage

func GetUserStorage() *UserStorage {
	if userStorage == nil {
		userStorage = &UserStorage{DB: GetPostgres()}
		if err := userStorage.Migrate(); err != nil {
			panic(err)
		}
	}
	return userStorage
}
