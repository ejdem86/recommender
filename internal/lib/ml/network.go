package ml

import (
	"bufio"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/Fontinalis/fonet"
	"github.com/ejdem86/recommender/errors"
)

type ActivationFunction string

const maxDepth = 10

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

type RoundFunc func(a float64, places int) float64

func RoundNearest(a float64, places int) float64 {
	multiplier := math.Pow10(places)
	return math.Round(a*multiplier) / multiplier
}

func RoundInteger(a float64, _ int) float64 {
	return math.Round(a)
}

func NoRound(a float64, _ int) float64 {
	return a
}

type Network interface {
	Export(w io.Writer) error
	MarshalJSON() ([]byte, error)
	Predict(input []float64) []float64
	Train(trainingData [][][]float64, epochs int, lrate float64, debug bool)
	UnmarshalJSON([]byte) error

	TrainFrom(src string, tp *TrainParams) error

	PredictVerified(in []float64, expected []float64, roundFunc RoundFunc, precision int) ([]float64, bool)
}

type N struct {
	tp *TrainParams

	*fonet.Network
}

type TrainParams struct {
	Epochs int
	Rate   float64
	Debug  bool
}

func (n *N) TrainFrom(src string, tp *TrainParams) error {
	trainingData, err := readTrainingData(src)
	if err != nil {
		return fmt.Errorf("failed to read the training data: %v", err)
	}

	s := trainingData.ToSlice()
	n.Train(s, tp.Epochs, tp.Rate, tp.Debug)
	n.tp = tp
	return nil
}

func (n *N) PredictVerified(in []float64, expected []float64, roundFunc RoundFunc, precision int) ([]float64, bool) {
	fmt.Println(in, expected)
	predictions := n.Predict(in)
	for i, v := range expected {
		expected[i] = roundFunc(v, precision)
	}
	fmt.Println(predictions, expected)
	for i, v := range predictions {
		if i > len(expected)-1 {
			break
		}
		predicted := roundFunc(v, precision)
		if math.Abs(expected[i]-predicted) > 0.00005 {
			go n.Train(TrainingData{TrainingSample{Input: in, Output: expected}}.ToSlice(), n.tp.Epochs, n.tp.Rate, true)
			return predictions, false
		}
	}
	return predictions, true
}

func (n *N) TrainNLevel(td TrainingData, tp *TrainParams, depth int) error {
	switch {
	case depth > maxDepth:
		return nil
	case depth == 0:
		n.tp = tp
	}
	firstPart := td[:len(td)/2]
	theRest := td[len(td)/2:]
	n.Train(firstPart.ToSlice(), tp.Epochs, tp.Rate, tp.Debug)

	var verifiedPredictions TrainingData
	for _, v := range theRest {
		_, isOK := n.PredictVerified(v.Input, v.Output, NoRound, 5)
		if isOK {
			verifiedPredictions = append(verifiedPredictions, v)
		}
	}
	return n.TrainNLevel(verifiedPredictions, tp, depth+1)
}

func NewNetwork(ls []int, activation ActivationFunction) (Network, error) {
	fn, err := fonet.NewNetwork(ls, functionMap[activation])
	if err != nil {
		return nil, fmt.Errorf("%w: %v", errors.ErrNetworkCreation, err)
	}
	return &N{Network: fn}, nil
}

func Load(source io.Reader, tp *TrainParams) (Network, error) {
	fn, err := fonet.Load(source)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", errors.ErrNetworkCreation, err)
	}
	return &N{Network: fn, tp: tp}, nil
}

func readTrainingData(f string) (TrainingData, error) {
	data, err := os.OpenFile(f, os.O_RDONLY, fs.ModePerm)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", errors.ErrReadFile, err)
	}
	defer data.Close()

	scanner := bufio.NewScanner(data)

	var td TrainingData
	for scanner.Scan() {
		line := strings.Split(scanner.Text(), " ")
		var ts TrainingSample
		for _, v := range strings.Split(line[0], ",") {
			value, err := strconv.ParseFloat(v, 64)
			if err != nil {
				return nil, fmt.Errorf("%w: %v", errors.ErrParseInput, err)
			}
			ts.Input = append(ts.Input, value)
		}
		for _, v := range strings.Split(line[1], ",") {
			value, err := strconv.ParseFloat(v, 64)
			if err != nil {
				return nil, fmt.Errorf("%w: %v", errors.ErrParseInput, err)
			}
			ts.Output = append(ts.Output, value)
		}
		td = append(td, ts)
	}

	return td, nil
}
