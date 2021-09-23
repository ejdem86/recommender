package main

import (
	"bufio"
	"fmt"
	"io/fs"
	"log"
	"os"
	"os/signal"
	"strconv"
	"strings"

	"github.com/caarlos0/env/v6"
	"github.com/ejdem86/recommender"
	"github.com/ejdem86/recommender/errors"
	"github.com/ejdem86/recommender/internal/lib/ml"
)

func main() {
	var cfg recommender.Config
	if err := env.Parse(&cfg); err != nil {
		log.Fatalf("Failed to parse the configuration: %v", err)
	}

	var network ml.Network
	var err error

	switch cfg.RestoreFrom {
	case "":
		network, err = trainNew(&cfg)
		if err != nil {
			log.Fatalf("training new network: %v", err)
		}
	default:
		network, err = restoreNetwork(cfg.RestoreFrom)
		if err != nil {
			log.Fatalf("restoring network: %v", err)
		}
	}

	closeChannel := make(chan os.Signal, 1)
	signal.Notify(closeChannel, os.Interrupt)
	go func() {
		<-closeChannel
		log.Println("Closing")
		output, err := os.Create("output.csv")
		if err != nil {
			log.Fatalf("Failed to open the output: %v", err)
		}
		if err = network.Export(output); err != nil {
			log.Fatalf("Failed to export the network: %v", err)
		}
		if err = output.Close(); err != nil {
			log.Fatalf("Failed to close the output file: %v", err)
		}
		os.Exit(0)
	}()

	user := bufio.NewReader(os.Stdin)
	for {
		tmp, err := user.ReadString('\n')
		if err != nil {
			log.Fatalf("Failed to read input: %v", err)
		}
		tmp = strings.ReplaceAll(tmp, "\n", "")
		predictRequest, err := parsePredictRequest(tmp)

		prediction := network.Predict(predictRequest)
		fmt.Printf("Predicted value: %v\n", prediction)
	}
}

func trainNew(cfg *recommender.Config) (ml.Network, error) {
	trainingData, err := readTrainingData(cfg.TrainingDataSource)
	if err != nil {
		log.Fatalf("Failed to read the training data: %v", err)
	}
	layersSetup := []int{cfg.InputLayers}
	layersSetup = append(layersSetup, cfg.HiddenLayers...)
	layersSetup = append(layersSetup, cfg.OutputLayers)
	network, err := ml.NewNetwork(layersSetup, cfg.ActivationMethod)
	if err != nil {
		return nil, fmt.Errorf("failed to create the network: %v", err)
	}

	s := trainingData.ToSlice()
	network.Train(s, cfg.Epochs, cfg.Rate, cfg.Debug)
	return network, nil
}

func restoreNetwork(source string) (ml.Network, error) {
	f, err := os.OpenFile(source, os.O_RDONLY, fs.ModePerm)
	if err != nil {
		return nil, fmt.Errorf("failed to open source network: %v", err)
	}
	defer f.Close()
	return ml.Load(f)
}

func parsePredictRequest(s string) ([]float64, error) {
	var ret []float64
	for _, v := range strings.Split(s, ",") {
		value, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return nil, fmt.Errorf("%w: %v", errors.ErrParseInput, err)
		}
		ret = append(ret, value)
	}
	if len(ret) == 0 {
		return nil, errors.ErrMissingInput
	}
	return ret, nil
}

func readTrainingData(f string) (ml.TrainingData, error) {
	data, err := os.OpenFile(f, os.O_RDONLY, fs.ModePerm)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", errors.ErrReadFile, err)
	}
	defer data.Close()

	scanner := bufio.NewScanner(data)

	var td ml.TrainingData
	for scanner.Scan() {
		line := strings.Split(scanner.Text(), " ")
		var ts ml.TrainingSample
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
