
package mlclient

import (
    "bytes"
    "crypto/hmac"
    "crypto/sha256"
    "encoding/hex"
    "io"
    "net/http"
    "os"
    "time"
)

func generateHMAC(message []byte, secret string) string {
    h := hmac.New(sha256.New, []byte(secret))
    h.Write(message)
    return hex.EncodeToString(h.Sum(nil))
}

func SendToML(jsonPayload []byte) (*http.Response, error) {
    mlEndpoint := os.Getenv("ML_ENDPOINT")
    secret := os.Getenv("HMAC_SECRET")
    if mlEndpoint == "" || secret == "" {
        return nil, ErrMissingConfig
    }

    signature := generateHMAC(jsonPayload, secret)

    req, err := http.NewRequest("POST", mlEndpoint, bytes.NewBuffer(jsonPayload))
    if err != nil {
        return nil, err
    }

    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("X-HMAC-SIGNATURE", signature)

    client := &http.Client{Timeout: 10 * time.Second}
    return client.Do(req)
}

var ErrMissingConfig = &ConfigError{"missing ML_ENDPOINT or HMAC_SECRET"}

type ConfigError struct {
    s string
}

func (e *ConfigError) Error() string {
    return e.s
}
