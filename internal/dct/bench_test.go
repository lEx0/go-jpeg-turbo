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
	"math/rand/v2"
	"runtime"
	"testing"

	"github.com/lEx0/go-jpeg-turbo/internal/cpuinfo"
)

// Kernel benchmarks: the *Ref specification (impl=ref), the portable Go
// kernel (impl=go) and the exported function at every SIMD level of the
// machine (impl=sse41, avx2, avx512 on amd64; impl=neon on arm64; see
// simdImpls) on the same inputs. Each iteration takes the next of 256
// prepared blocks. For the inverse DCT it fills the work block the way the
// decoder does, zeroing it and storing the non-zero coefficients one by one
// (so the kernels' loads meet those stores as they do in the decoder); the
// "prep" sub-benchmark measures that overhead alone.

const benchBlocks = 256

// simdImpl is a sub-benchmark of the exported function at a level.
type simdImpl struct {
	name  string
	level cpuinfo.Level
}

// simdImpls lists the levels the exported functions have kernels for: on
// amd64 every SIMD level of the machine, on arm64 NEON (at level Go), with
// -tags purego none (the exported function is the "go" kernel).
func simdImpls() []simdImpl {
	switch {
	case purego:
		return nil
	case runtime.GOARCH == "arm64":
		return []simdImpl{{"neon", cpuinfo.Go}}
	}
	var s []simdImpl
	for _, l := range cpuinfo.Levels()[1:] {
		s = append(s, simdImpl{l.String(), l})
	}
	return s
}

// runSIMD runs f as a sub-benchmark per entry of simdImpls, with the level
// set.
func runSIMD(b *testing.B, prefix string, f func(b *testing.B)) {
	for _, impl := range simdImpls() {
		b.Run(prefix+impl.name, func(b *testing.B) {
			prev := SetLevel(impl.level)
			defer SetLevel(prev)
			f(b)
		})
	}
}

func benchIDCTInputs(kind string) ([]Block, *Block) {
	r := rand.New(rand.NewPCG(42, 0))
	bs := make([]Block, benchBlocks)
	for i := range bs {
		switch kind {
		case "dc":
			genBlock(r, kindDC, &bs[i])
			bs[i][0] = int32(r.IntN(2048) - 1024)
		case "real":
			genBlock(r, kindReal, &bs[i])
		case "dense":
			for j := range bs[i] {
				bs[i][j] = int32(r.IntN(64) - 32)
			}
		case "mixed": // what a q75 photo decodes to: many DC-only blocks
			if r.IntN(3) == 0 {
				genBlock(r, kindDC, &bs[i])
				bs[i][0] = int32(r.IntN(2048) - 1024)
			} else {
				genBlock(r, kindReal, &bs[i])
			}
		}
	}
	var qt Block
	q := qualityTable(75, 0)
	for zig, v := range q {
		qt[Unzig[zig]] = int32(v)
	}
	return bs, &qt
}

var idctBenchKinds = []string{"dc", "real", "mixed", "dense"}

// coef is a non-zero coefficient of a block: natural index and value.
type coef struct {
	pos int32
	v   int32
}

// sparseBlocks lists the non-zero coefficients of each block, in zig-zag
// order (the order the decoder stores them in).
func sparseBlocks(bs []Block) [][]coef {
	cs := make([][]coef, len(bs))
	for i := range bs {
		for _, n := range Unzig {
			if v := bs[i][n]; v != 0 {
				cs[i] = append(cs[i], coef{int32(n), v})
			}
		}
	}
	return cs
}

// fill sets w to the block of the coefficients cs as the decoder does.
func fill(w *Block, cs []coef) {
	*w = Block{}
	for _, c := range cs {
		w[c.pos] = c.v
	}
}

func BenchmarkDequantIDCTStore(b *testing.B) {
	for _, kind := range idctBenchKinds {
		bs, qt := benchIDCTInputs(kind)
		dst := make([]byte, 64*8)
		cs := sparseBlocks(bs)
		run := func(f storeFn) func(b *testing.B) {
			return func(b *testing.B) {
				var w Block
				i := 0
				for b.Loop() {
					fill(&w, cs[i%benchBlocks])
					f(&w, qt, dst, 64)
					i++
				}
			}
		}
		b.Run(kind+"/impl=ref", run(DequantIDCTStoreRef))
		b.Run(kind+"/impl=go", run(dequantIDCTStoreGo))
		runSIMD(b, kind+"/impl=", run(DequantIDCTStore))
		b.Run(kind+"/impl=prep", run(func(b, qt *Block, dst []byte, stride int) {}))
	}
}

func BenchmarkDequantIDCT(b *testing.B) {
	for _, kind := range idctBenchKinds {
		bs, qt := benchIDCTInputs(kind)
		cs := sparseBlocks(bs)
		run := func(f func(b, qt *Block)) func(b *testing.B) {
			return func(b *testing.B) {
				var w Block
				i := 0
				for b.Loop() {
					fill(&w, cs[i%benchBlocks])
					f(&w, qt)
					i++
				}
			}
		}
		b.Run(kind+"/impl=ref", run(DequantIDCTRef))
		b.Run(kind+"/impl=go", run(dequantIDCTGo))
		runSIMD(b, kind+"/impl=", run(DequantIDCT))
	}
}

func BenchmarkFDCTQuant(b *testing.B) {
	r := rand.New(rand.NewPCG(43, 0))
	const stride = 64
	src := make([]byte, stride*8*benchBlocks/8+8)
	for i := range benchBlocks {
		var s [BlockSize]byte
		genSamples(r, []int{smpSmooth, smpNoise, smpRandom}[i%3], &s)
		x := (i % 8) * 8
		y := (i / 8) * 8
		for j := 0; j < 8; j++ {
			copy(src[(y+j)*stride+x:], s[8*j:8*j+8])
		}
	}
	q := qualityTable(75, 0)
	tab := newQuant(&q)
	run := func(f fdctFn) func(b *testing.B) {
		return func(b *testing.B) {
			var out [BlockSize]int16
			var nz uint64
			i := 0
			for b.Loop() {
				k := i % benchBlocks
				off := (k/8)*8*stride + (k%8)*8
				nz += f(src[off:], stride, tab, &out)
				i++
			}
			_ = nz
		}
	}
	b.Run("impl=ref", run(FDCTQuantRef))
	b.Run("impl=go", run(fdctQuantGo))
	runSIMD(b, "impl=", run(FDCTQuant))
}
