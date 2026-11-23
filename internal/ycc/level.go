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

import "github.com/lEx0/go-jpeg-turbo/internal/cpuinfo"

// level is the SIMD level selected for the kernels. On amd64 (without -tags
// purego) the wrappers in ycc_amd64.go select the kernels of the highest
// level at or below it that has kernels of its own; elsewhere the kernels
// are fixed by the build (NEON on arm64, portable Go otherwise) and level
// only records what SetLevel was given.
//
// Level reports level as selected, also when the kernels are those of a
// lower level: the AVX512 level runs the AVX2 kernels on a machine
// without AVX512 VNNI (ycc_amd64.go), and the jpeg tests rely on Level
// returning what SetLevel was given and on it matching internal/dct's.
// The test and benchmark names (implName) name the kernels that run.
var level = cpuinfo.Max

// Level reports the SIMD level the package's kernels use.
func Level() cpuinfo.Level { return level }

// SetLevel selects the kernels of level l for tests and benchmarks (clamped to cpuinfo.Max; a level without kernels of its own uses the next lower one) and returns the previous level. Not safe to call while kernels run.
func SetLevel(l cpuinfo.Level) cpuinfo.Level {
	prev := level
	level = max(cpuinfo.Go, min(l, cpuinfo.Max))
	return prev
}
