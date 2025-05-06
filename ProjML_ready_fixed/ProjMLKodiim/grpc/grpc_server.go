package grpc

import (
    "context"
    "errors"
    pb "projml/grpc/proto"
)

type Server struct {
    pb.UnimplementedUserServiceServer
}

func (s *Server) Register(ctx context.Context, req *pb.RegisterRequest) (*pb.AuthResponse, error) {
    if req.Login == "" || req.Password == "" {
        return nil, errors.New("login and password required")
    }
    // Имитация создания пользователя
    return &pb.AuthResponse{Token: "fake-jwt-for-" + req.Login}, nil
}

func (s *Server) SignIn(ctx context.Context, req *pb.SignInRequest) (*pb.AuthResponse, error) {
    if req.Login == "admin" && req.Password == "admin" {
        return &pb.AuthResponse{Token: "admin-token"}, nil
    }
    return nil, errors.New("invalid credentials")
}

func (s *Server) MyProfile(ctx context.Context, req *pb.ProfileRequest) (*pb.UserProfile, error) {
    return &pb.UserProfile{
        Login: "me",
        Email: "me@example.com",
    }, nil
}

func (s *Server) ProfileByLogin(ctx context.Context, req *pb.LoginRequest) (*pb.UserProfile, error) {
    if req.Login == "" {
        return nil, errors.New("login required")
    }
    return &pb.UserProfile{
        Login: req.Login,
        Email: req.Login + "@example.com",
    }, nil
}

func (s *Server) UpdatePassword(ctx context.Context, req *pb.UpdatePasswordRequest) (*pb.UpdateResponse, error) {
    if req.OldPassword == "" || req.NewPassword == "" {
        return nil, errors.New("passwords required")
    }
    return &pb.UpdateResponse{Success: true}, nil
}