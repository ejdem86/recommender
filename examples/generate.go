package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
)

var (
	intBool = map[bool]float64{
		true:  1,
		false: 0,
	}
)

func main() {
	target := os.Args[1]

	if _, err := os.Stat(target); err == nil {
		panic("target should not exist")
	}

	f, err := os.OpenFile(target, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		panic(err)
	}

	for i := 0; i < math.MaxInt16; i++ {
		randI := rand.Float64()
		randJ := rand.Float64()
		f.WriteString(fmt.Sprintf("%f,%f %d,%d\n", randI, randJ, intBool[randI > randJ], intBool[randI <= randJ]))
	}

	if err = f.Sync(); err != nil {
		fmt.Println(err)
	}
	if err = f.Close(); err != nil {
		fmt.Println(err)
	}
}
