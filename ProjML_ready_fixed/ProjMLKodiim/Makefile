PROTO_DIR=proto
PROTO_FILES=$(PROTO_DIR)/user.proto
OUT_DIR=proto

all: generate run

generate:
	protoc --go_out=$(OUT_DIR) --go-grpc_out=$(OUT_DIR) $(PROTO_FILES)

run:
	go run main.go

clean:
	rm -f $(OUT_DIR)/*.pb.go