package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/golang-jwt/jwt/v5"
	"github.com/gorilla/handlers"
	"golang.org/x/crypto/bcrypt"
	"log/slog"
	"net/http"
	"project/socnetwork"
	"project/storage"
	"strings"
	"time"
)

type Server struct {
	address string
	logger  *slog.Logger
	users   *socnetwork.UserRegistry
}

func NewServer(address string, logger *slog.Logger) *Server {
	userRegistry := socnetwork.NewUserRegistry(
		storage.GetUserStorage(),
	)
	return &Server{
		address: address,
		logger:  logger,
		users:   userRegistry,
	}
}

func (s *Server) Start() error {
	router := http.NewServeMux()

	router.HandleFunc("GET /api/ping", s.handlePing)
	router.HandleFunc("POST /api/auth/register", s.registerUser)
	router.HandleFunc("POST /api/auth/sign-in", s.signIn)
	router.HandleFunc("GET /api/me/profile", s.myProfile)
	router.HandleFunc("GET /api/profiles/{login}", s.profileByPhone)
	router.HandleFunc("POST /api/me/updatePassword", s.updatePassword)
	router.HandleFunc("POST /api/message", s.message)

	s.logger.Info("server has been started", "address", s.address)

	err := http.ListenAndServe(s.address, handlers.RecoveryHandler(handlers.PrintRecoveryStack(true))(router))
	if !errors.Is(err, http.ErrServerClosed) {
		return err
	}

	return nil
}

func (s *Server) handlePing(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}

func (s *Server) registerUser(w http.ResponseWriter, r *http.Request) {
	var user socnetwork.User
	if err := json.NewDecoder(r.Body).Decode(&user); err != nil {
		BadRequest(w, err.Error())
		return
	}

	if err := s.users.RegisterUser(&user); err != nil {
		if errors.Is(err, socnetwork.ErrEmailAlreadyExists) || errors.Is(err, socnetwork.ErrPhoneAlreadyExists) {
			LogicError(w, http.StatusConflict, err.Error())
			return
		}
		InternalError(w, s.logger, fmt.Errorf("не смог зарегестрировать пользователя: %w", err))
		return
	}

	type Response struct {
		Profile socnetwork.Profile `json:"profile"`
	}
	writeJSON(w, http.StatusCreated, &Response{Profile: user.Profile})
}

func (s *Server) signIn(w http.ResponseWriter, r *http.Request) {
	var credentials Credentials
	err := json.NewDecoder(r.Body).Decode(&credentials)
	if err != nil {
		BadRequest(w, err.Error())
		return
	}

	user, err := s.users.GetUserByPhone(credentials.Login)
	if err != nil {
		if errors.Is(err, socnetwork.ErrUserNotFound) {
			LogicError(w, http.StatusUnauthorized, err.Error())
			return
		}
		InternalError(w, s.logger, fmt.Errorf("не смог получить пользователя по логину: %w", err))
		return
	}

	if nil != bcrypt.CompareHashAndPassword(user.Password, []byte(credentials.Password)) {
		LogicError(w, http.StatusUnauthorized, "пароль не верен")
		return
	}

	expirationTime := time.Now().Add(12 * time.Hour)
	claims := &Claims{
		Login: credentials.Login,
		Tag:   string(user.Password),
		RegisteredClaims: jwt.RegisteredClaims{
			// In JWT, the expiry time is expressed as unix milliseconds
			ExpiresAt: jwt.NewNumericDate(expirationTime),
		},
	}

	// Конструктор токена вместе с указанием алгоритма подписи и полезной нагрузкой (claims)
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	// получаем подписанную строку
	tokenString, err := token.SignedString(JWTSecretKey)
	if err != nil {
		InternalError(w, s.logger, fmt.Errorf("не смог подписать токен: %w", err))
		return
	}

	type Response struct {
		Token string `json:"token"`
	}

	writeJSON(w, http.StatusOK, &Response{Token: tokenString})
}

func (s *Server) authByToken(r *http.Request) (*socnetwork.User, error) {
	if len(r.Header["Authorization"]) == 0 {
		return nil, errors.New("отсутствует токен авторизации")
	}
	token := r.Header["Authorization"][0]
	token = strings.TrimPrefix(token, "Bearer ")

	var claims Claims

	tkn, err := jwt.ParseWithClaims(token, &claims, func(token *jwt.Token) (any, error) {
		return JWTSecretKey, nil
	})
	if err != nil {
		if errors.Is(err, jwt.ErrSignatureInvalid) { // токен неправильно подписан, возможно подделка
			return nil, err
		}
		return nil, fmt.Errorf("не смог распарсить токен: %w", err)
	}
	if !tkn.Valid {
		return nil, errors.New("токен истек")
	}

	user, er := s.users.GetUserByPhone(claims.Login)
	if er != nil {
		return nil, fmt.Errorf("не смог получить пользователя указанного в токене: %w", er)
	}
	if claims.Tag != string(user.Password) {
		return nil, errors.New("пароль пользователя был изменен, токен более не валиден")
	}
	return user, nil
}

func (s *Server) myProfile(w http.ResponseWriter, r *http.Request) {
	user, err := s.authByToken(r)
	if err != nil {
		LogicError(w, http.StatusUnauthorized, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, &user)
}

func (s *Server) profileByPhone(w http.ResponseWriter, r *http.Request) {
	phone := r.PathValue("phone")

	profile, err := s.users.GetUserByPhone(phone)
	if err != nil {
		if errors.Is(err, socnetwork.ErrUserNotFound) {
			LogicError(w, http.StatusForbidden, "нет доступа к этому профилю")
			return
		}
		InternalError(w, s.logger, fmt.Errorf("не смог получить юзера по логину: %w", err))
		return
	}
	writeJSON(w, http.StatusOK, profile)
}

func (s *Server) updatePassword(w http.ResponseWriter, r *http.Request) {
	user, err := s.authByToken(r)
	if err != nil {
		LogicError(w, http.StatusUnauthorized, err.Error())
		return
	}

	type Request struct {
		OldPassword string
		NewPassword string
	}
	var request Request
	err = json.NewDecoder(r.Body).Decode(&request)
	if err != nil {
		BadRequest(w, err.Error())
		return
	}
	err = s.users.UpdatePassword(user, socnetwork.UpdatePasswordParams(request))
	if err != nil {
		if errors.Is(err, socnetwork.ErrInvalidPassword) {
			LogicError(w, http.StatusForbidden, socnetwork.ErrInvalidPassword.Error())
			return
		}
		if errors.Is(err, socnetwork.ErrMismatchPassword) {
			LogicError(w, http.StatusForbidden, socnetwork.ErrMismatchPassword.Error())
			return
		}
		InternalError(w, s.logger, err)
		return
	}

	type Response struct {
		Status string `json:"status"`
	}
	var response = Response{Status: "ok"}
	writeJSON(w, http.StatusOK, &response)
}

func (s *Server) message(w http.ResponseWriter, r *http.Request) {
	_, err := s.authByToken(r)
	if err != nil {
		LogicError(w, http.StatusUnauthorized, err.Error())
		return
	}

	type Request struct {
		Text string `json:"text"`
	}
	var request Request
	err = json.NewDecoder(r.Body).Decode(&request)
	if err != nil {
		BadRequest(w, err.Error())
		return
	}

	// TODO: при готовности ML разкомментить
	//mlResponse, err := socnetwork.CallML("http://localhost:5000/predict", request.Text)
	//if err != nil {
	//	InternalError(w, s.logger, fmt.Errorf("ошибка вызова ML: %w", err))
	//	return
	//}
	mlResponse := fmt.Sprintf("Заглушка ML: %s", request.Text)

	type Response struct {
		Result string `json:"result"`
	}
	writeJSON(w, http.StatusOK, &Response{Result: mlResponse})
}
