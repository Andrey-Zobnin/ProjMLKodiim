package socnetwork

import (
	"fmt"
	"bytes"
	"context"
	"io"
	"net/http"
	"encoding/json"
)

type MlService struct {
	PuthonEndpoint string
}

func Mlservice(endpoint string) *Mlservice {
	return &MlService{PythonEndpoint: endpoint}
}
