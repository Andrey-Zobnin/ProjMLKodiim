package storage

import (
	"database/sql"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jmoiron/sqlx"

	"project/socnetwork"
)

type UserStorage struct {
	DB *sqlx.DB
}

func (s *UserStorage) Migrate() error {
	queries := []string{
		`CREATE TABLE IF NOT EXISTS users (
			id serial PRIMARY KEY,
			firstName text,
			lastName text,
			password text,
			email text,
			phone text
		)`,
		`CREATE UNIQUE INDEX IF NOT EXISTS users_lower_email_uindex
			ON users (lower(email))`,
		`CREATE UNIQUE INDEX IF NOT EXISTS users_phone_uindex
			ON users (phone)`,
	}

	for _, q := range queries {
		if _, err := s.DB.Exec(q); err != nil {
			return fmt.Errorf("ошибка запроса (%q): %w", q, err)
		}
	}

	return nil
}

func (s *UserStorage) StoreUser(user *socnetwork.User) error {
	if user.ID != 0 {
		return s.updateProfile(user)
	}
	return s.insertUser(user)
}

func (s *UserStorage) insertUser(user *socnetwork.User) error {
	if len(user.Password) == 0 {
		panic("хеш пароля не может быть пустым")
	}

	const q = `INSERT INTO users (firstName, lastName, password, email, phone)
values ($1, $2, $3, $4, $5) returning id;`
	err := s.DB.Get(&user.ID, q, user.FirstName, user.LastName, user.Password, user.Email, user.Phone)
	if err != nil {
		var er *pgconn.PgError
		if errors.As(err, &er) {
			switch er.ConstraintName {
			case "users_lower(email)_uindex":
				return socnetwork.ErrEmailAlreadyExists
			case "users_phone_uindex":
				return socnetwork.ErrPhoneAlreadyExists
			}
		}
		return fmt.Errorf("ошибка записи новой строки: %w", err)
	}
	return nil
}

func (s *UserStorage) updateProfile(user *socnetwork.User) error {
	q := `UPDATE users SET 
    firstName = :firstName, 
    lastName = :lastName, 
    phone = :phone, 
    email = :email,
    password = COALESCE(:password, password)
	WHERE id = :id`

	_, err := s.DB.Exec(q, user.FirstName, user.LastName, user.Phone, user.Email, user.ID)
	if err != nil {
		return fmt.Errorf("ошибка запроса: %w", err)
	}

	// TODO: исправить запрос выше, чтобы делать update одним запросом
	if user.Password == nil {
		return nil
	}
	q = `UPDATE users SET  password=$1 WHERE id = $2`
	_, err = s.DB.Exec(q, user.Password, user.ID)
	if err != nil {
		return fmt.Errorf("ошибка запроса: %w", err)
	}
	return nil
}

func (s *UserStorage) FetchUser(user *socnetwork.User) error {
	q := `SELECT id, firstName, lastName, password, email, phone FROM users WHERE true `
	var args []any
	if user.ID != 0 {
		q += `AND id = ? `
		args = append(args, user.ID)
	}
	q = sqlx.Rebind(sqlx.DOLLAR, q)
	err := s.DB.QueryRowx(q, args...).Scan(&user.ID,
		&user.FirstName,
		&user.LastName,
		&user.Password,
		&user.Email,
		&user.Phone,
	)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return socnetwork.ErrUserNotFound
		}
		return fmt.Errorf("oшибка запроса к БД: %w", err)
	}
	return nil
}
