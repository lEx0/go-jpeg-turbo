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

package dct

import (
	"testing"

	"github.com/lEx0/go-jpeg-turbo/internal/cpuinfo"
)

// TestLevel pins the contract of Level and SetLevel that the jpeg tests
// and benchmarks rely on (internal/ycc has the same test): the default is
// cpuinfo.Max, SetLevel returns the previous level, and it clamps to
// [cpuinfo.Go, cpuinfo.Max], so that no request (an out-of-range value,
// or a level above what the machine supports) can select kernels the CPU
// lacks and raise SIGILL. From the level API check of the amd64 kernels'
// verification.
func TestLevel(t *testing.T) {
	if Level() != cpuinfo.Max {
		t.Fatalf("Level() = %v, want cpuinfo.Max %v by default", Level(), cpuinfo.Max)
	}
	defer SetLevel(SetLevel(cpuinfo.Max))
	for _, l := range []cpuinfo.Level{-5, -1, cpuinfo.Go, cpuinfo.SSE41, cpuinfo.AVX2, cpuinfo.AVX512, cpuinfo.AVX512 + 1, 100, cpuinfo.SSE41} {
		want := max(cpuinfo.Go, min(l, cpuinfo.Max))
		prev := Level()
		if got := SetLevel(l); got != prev {
			t.Errorf("SetLevel(%d) returned %v, want the previous level %v", l, got, prev)
		}
		if got := Level(); got != want {
			t.Errorf("after SetLevel(%d): Level() = %v, want %v", l, got, want)
		}
	}
}
