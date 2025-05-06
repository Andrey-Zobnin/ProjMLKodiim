package socnetwork

import (
	"errors"
	"fmt"
	"golang.org/x/crypto/bcrypt"
	"regexp"
	"time"
)

type Profile struct {
	Phone string `json:"phone"` // <= 20 characters \+[\d]+

	// ниже обновляемые поля
	Email     string `json:"email,omitempty"`     // [ 1 .. 50 ] characters
	FirstName string `json:"firstName,omitempty"` // [ 1 .. 50 ] characters
	LastName  string `json:"lastName,omitempty"`  // [ 1 .. 50 ] characters
	Birthday  string `json:"birthday,omitempty"`  // дд.мм.гггг
}

type User struct {
	ID int `json:"-"`
	Profile
	/* RawPassword используется только для регистрации данных
	Длина пароля не менее 6 символов.
	Присутствуют латинские символы в нижнем и верхнем регистре.
	Присутствует минимум одна цифра.
	*/
	RawPassword string `json:"password"` // [ 6 .. 100 ] characters
	Password    []byte `json:"-"`        // bcrypt hashed
}

type UserStorer interface {
	// StoreUser сохраняет нового или обновляет старого пользователя
	// обновление происходит если присутсвует ID
	// возможные ошибки: ErrLoginAlreadyExists, ErrEmailAlreadyExists, ErrPhoneAlreadyExists
	StoreUser(*User) error
}

type UserFetcher interface {
	// FetchUser получает пользователя по ID или логину
	// возможные ошибки: ErrUserNotFound
	FetchUser(*User) error
}

type UserRepository interface {
	UserStorer
	UserFetcher
}

var (
	ErrEmailAlreadyExists = errors.New("пользователь с такой почтой уже существует")
	ErrPhoneAlreadyExists = errors.New("пользователь с таким телефоном уже существует")
	ErrInvalidUser        = errors.New("данные пользователя содержат ошибки")
	ErrUserNotFound       = errors.New("пользователь не найден")
)

type UserRegistry struct {
	users UserRepository
}

func NewUserRegistry(users UserRepository) *UserRegistry {
	return &UserRegistry{users: users}
}

// RegisterUser валидирует значения полей пользователя и сохраняет его
// При сохранении нового пользователя назначает ему идентификатор.
// возможные ошибки: ErrLoginAlreadyExists, ErrEmailAlreadyExists, ErrPhoneAlreadyExists, ErrInvalidUser
func (r *UserRegistry) RegisterUser(user *User) error {
	if !IsValidUser(user) {
		return ErrInvalidUser
	}

	user.Password, _ = bcrypt.GenerateFromPassword([]byte(user.RawPassword), 12)

	if err := r.users.StoreUser(user); err != nil {
		if errors.Is(err, ErrEmailAlreadyExists) || errors.Is(err, ErrPhoneAlreadyExists) {
			return err
		}
		return fmt.Errorf("не смог сохранить пользователя: %w", err)
	}
	return nil
}

func IsValidUser(user *User) bool {
	return IsValidFirstName(user.FirstName) &&
		IsValidSecondName(user.LastName) &&
		IsValidEmail(user.Email) &&
		IsValidPhone(user.Phone) &&
		IsValidDate(user.Birthday) &&
		IsValidRawPassword(user.RawPassword)
}

// IsValidRawPassword проверяет исходный пароль по критериям:
//
//	 Длина пароля не менее 6 символов.
//		Присутствуют латинские символы в нижнем и верхнем регистре.
//		Присутствует минимум одна цифра.
func IsValidRawPassword(value string) bool {
	if len(value) < 6 {
		return false
	}
	var (
		hasDigit          bool
		hasUpperLatinCase bool
		hasLowerLatinCase bool
	)
	for _, r := range value {
		if r >= 'a' && r <= 'z' {
			hasLowerLatinCase = true
			continue
		}
		if r >= 'A' && r <= 'Z' {
			hasUpperLatinCase = true
			continue
		}
		if r >= '0' && r <= '9' {
			hasDigit = true
		}
	}
	return hasDigit && hasLowerLatinCase && hasUpperLatinCase
}

func IsValidEmail(email string) bool {
	return len(email) <= 50 && len(email) > 0
}

func IsValidFirstName(firstName string) bool {
	return len(firstName) <= 50 && len(firstName) > 1
}

func IsValidSecondName(secondName string) bool {
	return len(secondName) <= 50 && len(secondName) > 1
}

var validPhone = regexp.MustCompile(`^\+\d{1,19}$`)

func IsValidPhone(phone string) bool {
	return phone == "" || validPhone.MatchString(phone)
}

func IsValidDate(birthday string) bool {
	if birthday == "" {
		return true
	}

	_, err := time.Parse("02.01.2006", birthday)
	return err == nil
}

func (r *UserRegistry) GetUserByPhone(phone string) (*User, error) {
	user := User{Profile: Profile{
		Phone: phone,
	}}

	if err := r.users.FetchUser(&user); err != nil {
		return nil, fmt.Errorf("не смог получить пользователя '%s': %w", phone, err)
	}
	return &user, nil
}

type UpdatePasswordParams struct {
	OldPassword string
	NewPassword string
}

var ErrInvalidPassword = errors.New("пароль не валидный")
var ErrMismatchPassword = errors.New("пароли не совпадают")

func (r *UserRegistry) UpdatePassword(user *User, params UpdatePasswordParams) error {
	if nil != bcrypt.CompareHashAndPassword(user.Password, []byte(params.OldPassword)) {
		return ErrMismatchPassword
	}
	if !IsValidRawPassword(params.NewPassword) {
		return ErrInvalidPassword
	}
	user.RawPassword = params.NewPassword
	user.Password, _ = bcrypt.GenerateFromPassword([]byte(user.RawPassword), 12)
	err := r.users.StoreUser(user)
	if err != nil {
		return fmt.Errorf("не смог сохранить пользователя: %w", err)
	}
	return nil
}
