package main

import (
    "context"
    "log"
    "time"

    "google.golang.org/grpc"
    pb "projml/grpc/proto"
)

func main() {
    conn, err := grpc.Dial("localhost:50051", grpc.WithInsecure())
    if err != nil {
        log.Fatalf("did not connect: %v", err)
    }
    defer conn.Close()
    client := pb.NewUserServiceClient(conn)

    ctx, cancel := context.WithTimeout(context.Background(), time.Second)
    defer cancel()

    // Register
    res1, err := client.Register(ctx, &pb.RegisterRequest{Login: "testuser", Password: "pass123"})
    if err != nil {
        log.Fatalf("could not register: %v", err)
    }
    log.Printf("Register Token: %s", res1.Token)

    // SignIn
    res2, err := client.SignIn(ctx, &pb.SignInRequest{Login: "admin", Password: "admin"})
    if err != nil {
        log.Fatalf("could not sign in: %v", err)
    }
    log.Printf("SignIn Token: %s", res2.Token)

    // MyProfile
    res3, err := client.MyProfile(ctx, &pb.ProfileRequest{})
    if err != nil {
        log.Fatalf("could not get profile: %v", err)
    }
    log.Printf("My Profile: %+v", res3)

    // ProfileByLogin
    res4, err := client.ProfileByLogin(ctx, &pb.LoginRequest{Login: "testuser"})
    if err != nil {
        log.Fatalf("could not get profile by login: %v", err)
    }
    log.Printf("Profile By Login: %+v", res4)

    // UpdatePassword
    res5, err := client.UpdatePassword(ctx, &pb.UpdatePasswordRequest{OldPassword: "old", NewPassword: "new"})
    if err != nil {
        log.Fatalf("could not update password: %v", err)
    }
    log.Printf("Update Success: %v", res5.Success)
}