FROM scratch
COPY --from=builder /app/backend /backend
ENTRYPOINT ["/backend"]

FROM golang:1.23-alpine AS builder
WORKDIR /app
COPY ProjMLKodiim .
RUN go mod download
RUN CGO_ENABLED=0 GOOS=linux go build -o backend .

FROM scratch
COPY --from=builder /app/backend /backend
ENTRYPOINT ["/backend"]
