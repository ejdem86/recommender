package recommender

import "github.com/ejdem86/recommender/internal/lib/ml"

// Config is the configuration for the service
type Config struct {
	Debug       bool   `env:"DEBUG" envDefault:"false"`
	RestoreFrom string `env:"RESTORE_FROM" envDefault:""`
	PersistTo   string `env:"PERSIST_TO" envDefault:"networks/output.network"`

	NetworkConfig
	LearnConfig
}

type NetworkConfig struct {
	InputLayers      int                   `env:"INPUT_LAYERS" envDefault:"2"`
	HiddenLayers     []int                 `env:"HIDDEN_LAYERS" envDefault:"20,20"`
	OutputLayers     int                   `env:"OUTPUT_LAYERS" envDefault:"2"`
	ActivationMethod ml.ActivationFunction `env:"ACTIVATION_METHOD" envDefault:"ReLU"`
}

type LearnConfig struct {
	TrainingDataSource string  `env:"TRAINING_DATA_SOURCE"`
	Epochs             int     `env:"TRAINING_EPOCHS" envDefault:"10"`
	Rate               float64 `env:"TRAINING_RATE" envDefault:"1.2"`
	NLevel             bool    `env:"N_LEVEL" envDefault:"true"`
}
