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
	"time"

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
		network, err = restoreNetwork(&cfg)
		if err != nil {
			log.Fatalf("restoring network: %v", err)
		}

		if cfg.TrainingDataSource != "" {
			if err = network.TrainFrom(cfg.TrainingDataSource, &ml.TrainParams{
				Epochs: cfg.Epochs,
				Rate:   cfg.Rate,
				Debug:  cfg.Debug,
			}); err != nil {
				log.Fatalf("training network: %v", err)
			}
		}
	}

	closeChannel := make(chan os.Signal, 1)
	signal.Notify(closeChannel, os.Interrupt)
	go func() {
		<-closeChannel
		log.Println("Closing")
		if _, err = os.Stat(cfg.PersistTo); err == nil {
			if err = os.Rename(cfg.PersistTo, fmt.Sprintf("%s-%d.network", strings.Split(cfg.PersistTo, ".")[0], time.Now().Unix())); err != nil {
				log.Println("Failed to move old network to a new file:", err)
			}
		}
		output, err := os.Create(cfg.PersistTo)
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
		predictRequest, expected, err := parsePredictRequest(tmp)

		prediction, isOK := network.PredictVerified(predictRequest, expected, ml.RoundNearest, 3)
		fmt.Printf("Predicted value: %v, matches: %v\n", prediction, isOK)
	}
}

func trainNew(cfg *recommender.Config) (ml.Network, error) {
	layersSetup := []int{cfg.InputLayers}
	layersSetup = append(layersSetup, cfg.HiddenLayers...)
	layersSetup = append(layersSetup, cfg.OutputLayers)
	network, err := ml.NewNetwork(layersSetup, cfg.ActivationMethod)
	if err != nil {
		return nil, fmt.Errorf("failed to create the network: %v", err)
	}
	if err = network.TrainFrom(cfg.TrainingDataSource, &ml.TrainParams{
		Epochs: cfg.Epochs,
		Rate:   cfg.Rate,
		Debug:  cfg.Debug,
	}); err != nil {
		return nil, fmt.Errorf("failed to train the new network: %v", err)
	}
	return network, nil
}

func restoreNetwork(cfg *recommender.Config) (ml.Network, error) {
	f, err := os.OpenFile(cfg.RestoreFrom, os.O_RDONLY, fs.ModePerm)
	if err != nil {
		return nil, fmt.Errorf("failed to open source network: %v", err)
	}
	defer f.Close()
	return ml.Load(f, &ml.TrainParams{
		Epochs: cfg.Epochs,
		Rate:   cfg.Rate,
	})
}

func parsePredictRequest(s string) ([]float64, []float64, error) {
	inputS := strings.Split(s, " ")
	switch len(inputS) {
	case 1:
		ret, err := parseFloatSlice(inputS[0])
		if err != nil {
			return nil, nil, err
		}
		if len(ret) == 0 {
			return nil, nil, errors.ErrMissingInput
		}
		return ret, nil, nil
	case 2:
		dataToPredict, err := parseFloatSlice(inputS[0])
		if err != nil {
			return nil, nil, err
		}
		dataToCompare, err := parseFloatSlice(inputS[1])
		if err != nil {
			return nil, nil, err
		}
		return dataToPredict, dataToCompare, nil
	default:
		return nil, nil, errors.ErrParseInput
	}
}

func parseFloatSlice(s string) ([]float64, error) {
	var ret []float64
	for _, v := range strings.Split(s, ",") {
		value, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return nil, fmt.Errorf("%w: %v", errors.ErrParseInput, err)
		}
		ret = append(ret, value)
	}
	return ret, nil
}
