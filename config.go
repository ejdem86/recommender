package recommender

import "github.com/ejdem86/recommender/internal/lib/ml"

// Config is the configuration for the service
type Config struct {
	Debug       bool   `env:"DEBUG" envDefault:"true"`
	RestoreFrom string `env:"RESTORE_FROM" envDefault:""`

	NetworkConfig
	LearnConfig
}

type NetworkConfig struct {
	InputLayers      int                   `env:"INPUT_LAYERS" envDefault:"2"`
	HiddenLayers     []int                 `env:"HIDDEN_LAYERS" envDefault:"5,5"`
	OutputLayers     int                   `env:"OUTPUT_LAYERS" envDefault:"1"`
	ActivationMethod ml.ActivationFunction `env:"ACTIVATION_METHOD" envDefault:"Sigmond"`
}

type LearnConfig struct {
	TrainingDataSource string  `env:"TRAINING_DATA_SOURCE,required"`
	Epochs             int     `env:"TRAINING_EPOCHS" envDefault:"100000"`
	Rate               float64 `env:"TRAINING_RATE" envDefault:"1.1"`
}
