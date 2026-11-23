// The MIT License (MIT)
//
// Copyright (c) 2026 Amangeldy Kadyl
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

package jpeg

import (
	"flag"
	"testing"
)

// raceShort makes the race detector imply -short unless -short was given
// explicitly (go test -race -short=false runs the full sweeps); TestMain
// calls it after flag.Parse. The input sweeps of the compatibility tests
// (truncations, mutations, buffer boundaries, reader errors, encoder
// inputs) are what takes time, and the race detector makes them about 8
// times slower: on a shared EPYC 8124P the package took 27 to 39 minutes
// with -race and about 5 without, and takes about 4 with -race now. The
// decoder and the encoder run on the calling goroutine (TestEncConcurrent,
// which does not depend on -short, covers concurrent use), so the race
// detector gains little from the full sweeps over the -short ones; the
// full sweeps belong to the runs without -race.
func raceShort() error {
	flag.Visit(func(f *flag.Flag) { shortSet = shortSet || f.Name == "test.short" })
	if !encRaceEnabled || shortSet {
		return nil
	}
	return flag.Set("test.short", "true")
}

// shortSet reports that -short was given on the command line.
var shortSet bool

// TestRaceShort checks that the race detector implies -short (raceShort).
func TestRaceShort(t *testing.T) {
	if encRaceEnabled && !shortSet && !testing.Short() {
		t.Fatal("-race without -short: the sweeps run at full size (27 to 39 minutes); see raceShort")
	}
	t.Logf("race %v, -short given %v, testing.Short() %v", encRaceEnabled, shortSet, testing.Short())
}
