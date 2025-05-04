package crypto

import (
    "crypto/aes"
    "crypto/cipher"
    "crypto/rand"
    "encoding/base64"
    "errors"
    "fmt"
    "io"
)

// TODO уменьшить дублирование кода 

func check[T any](v T, err error) (T, error) {
    if err != nil {
        var zero T
        return zero, err
    }
    return v, nil
}

func EncryptAESGCM(plaintext, keyB64 string) (string, error) {
    key, err := base64.StdEncoding.DecodeString(keyB64)
    if err != nil {
        return "", err
    }

    block, err := check(aes.NewCipher(key))
    if err != nil {
        return "", err
    }

    aesgcm, err := check(cipher.NewGCM(block))
    if err != nil {
        return "", err
    }

    nonce := make([]byte, aesgcm.NonceSize())
    if _, err = io.ReadFull(rand.Reader, nonce); err != nil {
        return "", err
    }

    ciphertext := aesgcm.Seal(nonce, nonce, []byte(plaintext), nil)
    return base64.StdEncoding.EncodeToString(ciphertext), nil
}

func DecryptAESGCM(ciphertextB64, keyB64 string) (string, error) {
    ciphertext, err := check(base64.StdEncoding.DecodeString(ciphertextB64))
    if err != nil {
        return "", err
    }

    key, err := check(base64.StdEncoding.DecodeString(keyB64))
    if err != nil {
        return "", err
    }

    block, err := check(aes.NewCipher(key))
    if err != nil {
        return "", err
    }

    aesgcm, err := check(cipher.NewGCM(block))
    if err != nil {
        return "", err
    }

    nonceSize := aesgcm.NonceSize()
    if len(ciphertext) < nonceSize {
        return "", errors.New("ciphertext too short")
    }

    nonce, ciphertext := ciphertext[:nonceSize], ciphertext[nonceSize:]
    plaintext, err := aesgcm.Open(nil, nonce, ciphertext, nil)
    if err != nil {
        return "", err
    }

    return string(plaintext), nil
}
