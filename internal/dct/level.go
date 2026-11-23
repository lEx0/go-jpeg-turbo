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

import "github.com/lEx0/go-jpeg-turbo/internal/cpuinfo"

// level is the SIMD level of the amd64 kernels (see dct_amd64.go); the
// other builds only record it.
var level = cpuinfo.Max

// Level reports the SIMD level the package's kernels use.
func Level() cpuinfo.Level { return level }

// SetLevel selects the kernels of level l for tests and benchmarks (clamped
// to cpuinfo.Max; a level without kernels of its own uses the next lower
// one) and returns the previous level. Not safe to call while kernels run.
func SetLevel(l cpuinfo.Level) cpuinfo.Level {
	prev := level
	level = min(max(l, cpuinfo.Go), cpuinfo.Max)
	return prev
}
