package main

import (
    "log"
    "net"

    "google.golang.org/grpc"
    pb "projml/grpc/proto"
    "projml/grpc/grpc"
)

func main() {
    lis, err := net.Listen("tcp", ":50051")
    if err != nil {
        log.Fatalf("failed to listen: %v", err)
    }
    s := grpc.NewServer()
    pb.RegisterUserServiceServer(s, &grpc.Server{})
    log.Println("gRPC server listening on :50051")
    if err := s.Serve(lis); err != nil {
        log.Fatalf("failed to serve: %v", err)
    }
}