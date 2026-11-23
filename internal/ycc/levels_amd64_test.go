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

//go:build amd64 && !purego

package ycc

import (
	"fmt"
	"path"
	"slices"
	"testing"

	"github.com/lEx0/go-jpeg-turbo/internal/cpuinfo"
)

// testLevels returns the levels the tests run the kernels at: every level
// this machine supports that has kernels of its own (not the AVX512 level
// on a machine without AVX512 VNNI, which runs the AVX2 kernels).
func testLevels() []cpuinfo.Level {
	var ls []cpuinfo.Level
	for _, l := range cpuinfo.Levels() {
		if kernelsOf(l) == l {
			ls = append(ls, l)
		}
	}
	return ls
}

// TestKernelLevel checks which kernels the wrappers pick for each level
// and row length: the highest level at or below the selected one that
// has kernels and whose block the row fills.
func TestKernelLevel(t *testing.T) {
	defer SetLevel(SetLevel(cpuinfo.Max))
	block := map[cpuinfo.Level]int{cpuinfo.SSE41: 16, cpuinfo.AVX2: 32, cpuinfo.AVX512: 64}
	for _, l := range cpuinfo.Levels() {
		SetLevel(l)
		for n := -1; n <= 200; n++ {
			want := cpuinfo.Go
			for k := cpuinfo.SSE41; k <= l; k++ {
				if n >= block[k] && (k != cpuinfo.AVX512 || hasAVX512Kernels) {
					want = k
				}
			}
			if got := kernelLevel(n); got != want {
				t.Fatalf("level %v: kernelLevel(%d) = %v, want %v", l, n, got, want)
			}
		}
	}
}

// TestAVX512Fallback checks that the AVX512 level runs the AVX2 kernels
// on a machine without AVX512 VNNI.
func TestAVX512Fallback(t *testing.T) {
	if cpuinfo.Max < cpuinfo.AVX512 {
		t.Skip("no AVX512 level")
	}
	defer SetLevel(SetLevel(cpuinfo.AVX512))
	defer func(saved bool) { hasAVX512Kernels = saved }(hasAVX512Kernels)
	hasAVX512Kernels = false
	if got := kernelLevel(1000); got != cpuinfo.AVX2 {
		t.Fatalf("kernelLevel(1000) = %v without AVX512 VNNI, want avx2", got)
	}
	testRows(t)
	testRows420(t)
	testDownsample2x2Rows(t)
}

// TestRegressFallbackName: on an AVX-512 machine without AVX512 VNNI
// (Skylake-SP/X) the AVX512 level runs the AVX2 kernels, but implName,
// and with it every test and benchmark name, said avx512, and
// forEachLevel and benchImpls ran the AVX2 kernels a second time under
// that name. Such a machine is simulated by clearing hasAVX512Kernels
// (and, on a machine with VNNI, the real case is checked too).
func TestRegressFallbackName(t *testing.T) {
	if cpuinfo.Max < cpuinfo.AVX512 {
		t.Skip("no AVX512 level")
	}
	defer SetLevel(SetLevel(cpuinfo.AVX512))
	defer func(saved bool) { hasAVX512Kernels = saved }(hasAVX512Kernels)
	vnni := []bool{false}
	if cpuinfo.X86.HasAVX512VNNI {
		vnni = append(vnni, true)
	}
	for _, v := range vnni {
		t.Run(fmt.Sprintf("vnni=%v", v), func(t *testing.T) {
			hasAVX512Kernels = v
			SetLevel(cpuinfo.AVX512)
			want, levels := "avx512", cpuinfo.Levels()
			if !v {
				want, levels = "avx2", levels[:len(levels)-1]
			}
			if got := implName(); got != want {
				t.Errorf("implName() = %s at the AVX512 level, want %s", got, want)
			}
			if got := testLevels(); !slices.Equal(got, levels) {
				t.Errorf("testLevels() = %v, want %v", got, levels)
			}
			// The subtests and benchmarks are named after the kernels
			// that run, each set once.
			var names, wantNames []string
			forEachLevel(t, func(t *testing.T) { names = append(names, path.Base(t.Name())) })
			for _, l := range levels {
				wantNames = append(wantNames, l.String())
			}
			if !slices.Equal(names, wantNames) {
				t.Errorf("forEachLevel subtests %v, want %v", names, wantNames)
			}
			if Level() != cpuinfo.AVX512 {
				t.Errorf("Level() = %v after SetLevel(avx512), want avx512 (the level selected)", Level())
			}
		})
	}
}
