package ml

import (
	"fmt"
	"io"

	"github.com/Fontinalis/fonet"
	"github.com/ejdem86/recommender/errors"
)

type ActivationFunction string

var (
	Sigmond      = ActivationFunction(fonet.Sigmond.String())
	BentIdentity = ActivationFunction(fonet.BentIdentity.String())
	ReLU         = ActivationFunction(fonet.ReLU.String())
	LeakyReLU    = ActivationFunction(fonet.LeakyReLU.String())
	ArSinH       = ActivationFunction(fonet.ArSinH.String())

	functionMap = map[ActivationFunction]fonet.ActivationFunction{
		Sigmond:      fonet.Sigmond,
		BentIdentity: fonet.BentIdentity,
		ReLU:         fonet.ReLU,
		LeakyReLU:    fonet.LeakyReLU,
		ArSinH:       fonet.ArSinH,
	}
)

type TrainingSample struct {
	Input  []float64
	Output []float64
}

func (ts TrainingSample) ToSlice() (result [][]float64) {
	result = append(result, ts.Input)
	result = append(result, ts.Output)
	return
}

type TrainingData []TrainingSample

func (td TrainingData) ToSlice() (result [][][]float64) {
	for _, v := range td {
		result = append(result, v.ToSlice())
	}
	return
}

type Network interface {
	Export(w io.Writer) error
	MarshalJSON() ([]byte, error)
	Predict(input []float64) []float64
	Train(trainingData [][][]float64, epochs int, lrate float64, debug bool)
	UnmarshalJSON([]byte) error
}

func NewNetwork(ls []int, activation ActivationFunction) (Network, error) {
	fn, err := fonet.NewNetwork(ls, functionMap[activation])
	if err != nil {
		return nil, fmt.Errorf("%w: %v", errors.ErrNetworkCreation, err)
	}
	return fn, nil
}

func Load(source io.Reader) (Network, error) {
	fn, err := fonet.Load(source)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", errors.ErrNetworkCreation, err)
	}
	return fn, nil
}
