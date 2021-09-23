package errors

import "errors"

var (
	ErrNetworkCreation = Error{errors.New("failed to create network")}
	ErrReadFile        = Error{errors.New("failed to read data")}
	ErrParseInput      = Error{errors.New("failed to parse input")}
	ErrMissingInput    = Error{errors.New("missing input, provide a valid input")}
)

type Error struct {
	error
}

func (e Error) Error() string {
	if e.error == nil {
		return ""
	}
	return e.error.Error()
}
