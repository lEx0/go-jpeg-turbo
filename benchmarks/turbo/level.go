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

package turbo

// The library picks its SIMD level once, from the CPU, and has no knob for
// it. The benchmarks select a lower level the way the jpeg package's
// tests do (jpeg/level_test.go): through the internal packages, which this
// module may import because its import path lies under the repository's.
// run.sh's builds turbo-sse41, turbo-avx2 and so on set the variable.

import (
	"fmt"
	"os"
	"runtime"

	"github.com/lEx0/go-jpeg-turbo/internal/cpuinfo"
	"github.com/lEx0/go-jpeg-turbo/internal/dct"
	"github.com/lEx0/go-jpeg-turbo/internal/ycc"
)

// levelEnv is the environment variable that selects the kernel level.
const levelEnv = "JPEG_TURBO_LEVEL"

func init() {
	if err := setLevel(os.Getenv(levelEnv)); err != nil {
		fmt.Fprintln(os.Stderr, "benchmarks/turbo:", err)
		os.Exit(2)
	}
}

// setLevel selects level s (empty: keep the CPU's best) in both kernel
// packages. As in the jpeg tests, on arm64 and with -tags purego the
// kernels are fixed by the build and a level is only recorded.
func setLevel(s string) error {
	if s == "" {
		return nil
	}
	l, err := cpuinfo.ParseLevel(s)
	if err != nil {
		return fmt.Errorf("%s=%s: %v", levelEnv, s, err)
	}
	if l > cpuinfo.Max {
		return fmt.Errorf("%s=%s: this machine (%s) supports up to %v", levelEnv, s, runtime.GOARCH, cpuinfo.Max)
	}
	dct.SetLevel(l)
	ycc.SetLevel(l)
	return nil
}
