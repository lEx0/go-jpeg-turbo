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

package ycc

import (
	"testing"

	"github.com/lEx0/go-jpeg-turbo/internal/cpuinfo"
)

// forEachLevel runs f as a subtest at every level of testLevels, named
// after the kernels in use (go, sse41, avx2, avx512 on amd64; neon on
// arm64), with the level restored afterwards.
func forEachLevel(t *testing.T, f func(t *testing.T)) {
	t.Helper()
	for _, l := range testLevels() {
		func() {
			defer SetLevel(SetLevel(l))
			t.Run(implName(), f)
		}()
	}
}

func TestLevel(t *testing.T) {
	if Level() != cpuinfo.Max {
		t.Fatalf("Level() = %v, want cpuinfo.Max %v by default", Level(), cpuinfo.Max)
	}
	defer SetLevel(SetLevel(cpuinfo.Max))
	for _, l := range []cpuinfo.Level{cpuinfo.Go, cpuinfo.SSE41, cpuinfo.AVX2, cpuinfo.AVX512, cpuinfo.AVX512 + 1, -1} {
		want := max(cpuinfo.Go, min(l, cpuinfo.Max))
		prev := Level()
		if got := SetLevel(l); got != prev {
			t.Errorf("SetLevel(%v) returned %v, want the previous level %v", l, got, prev)
		}
		if got := Level(); got != want {
			t.Errorf("after SetLevel(%v): Level() = %v, want %v", l, got, want)
		}
	}
	t.Logf("levels %v, kernels %s at Max", testLevels(), func() string {
		defer SetLevel(SetLevel(cpuinfo.Max))
		return implName()
	}())
}
