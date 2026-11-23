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
	"math"
	"math/big"
	"math/rand/v2"
	"testing"
)

// TestFDCTBound proves the bound the NEON FDCT relies on: for every block
// of uint8 samples, every value fdctCols and fdctRows compute stays within
// int32 (so Go's arithmetic never wraps), and every fdctRows output v
// before the final cut satisfies -2^31 <= v < 2^31 - 2^17 (so
// (v + 1<<17) >> 18 does not wrap either and equals SRSHR(v>>16, #2)).
//
// The proof is affine arithmetic: each value is tracked as an exact linear
// form in the 64 samples s_i in [0, 255] plus an error interval. Additions,
// subtractions and constant multiplications are exact on the forms; an
// arithmetic shift right by n of an integer x is x/2^n - f with f in
// [0, 1 - 2^-n], which widens the error interval. The range of a form is
// then exact for its linear part (each s_i at 0 or 255 independently) plus
// the error interval, a rigorous bound for all 256^64 blocks.
func TestFDCTBound(t *testing.T) {
	var b fblock
	for i := range b {
		b[i] = fsample(i)
	}
	minLo, maxHi := math.Inf(1), math.Inf(-1)
	pre := fdctForms(t, &b)
	for i, v := range pre {
		lo, hi := v.bounds()
		if lo.Cmp(big.NewRat(math.MinInt32, 1)) < 0 || hi.Cmp(big.NewRat(math.MaxInt32-1<<17+1, 1)) >= 0 {
			t.Errorf("output %d: [%s, %s] outside [-2^31, 2^31-2^17)", i, lo.FloatString(1), hi.FloatString(1))
		}
		lf, _ := lo.Float64()
		hf, _ := hi.Float64()
		minLo, maxHi = min(minLo, lf), max(maxHi, hf)
	}
	t.Logf("fdctRows outputs in [%.2f, %.2f] * 2^18 (need [-8192, 8191.5) * 2^18)", minLo/(1<<18), maxHi/(1<<18))

	// The forms describe the actual computation: check random blocks (in
	// float64, which is only a consistency check of the replica below).
	type flt struct {
		c        [BlockSize]float64
		c0       float64
		elo, ehi float64
	}
	var cuts [BlockSize]flt
	for i, v := range pre {
		cut := v.add(fconst(1 << 17)).sshr(18)
		for j, c := range cut.c {
			cuts[i].c[j], _ = c.Float64()
		}
		cuts[i].c0, _ = cut.c0.Float64()
		cuts[i].elo, _ = cut.elo.Float64()
		cuts[i].ehi, _ = cut.ehi.Float64()
	}
	r := rand.New(rand.NewPCG(15, 16))
	for n := 0; n < 20000; n++ {
		var s [BlockSize]byte
		genSamples(r, n%numSmp, &s)
		var x Block
		for i, v := range s {
			x[i] = int32(v)
		}
		FDCT(&x)
		for i, f := range cuts {
			v := f.c0
			for j, c := range f.c {
				v += c * float64(s[j])
			}
			if got := float64(x[i]); got < v+f.elo-1e-6 || got > v+f.ehi+1e-6 {
				t.Fatalf("block %v: output %d = %d outside its form's [%f, %f]", s, i, x[i], v+f.elo, v+f.ehi)
			}
		}
	}
}

// form is c0 + sum(c[i]*s[i]) + e, e in [elo, ehi].
type form struct {
	c        [BlockSize]*big.Rat
	c0       *big.Rat
	elo, ehi *big.Rat
}

type fblock [BlockSize]form

func fconst(v int64) form {
	var f form
	for i := range f.c {
		f.c[i] = new(big.Rat)
	}
	f.c0 = big.NewRat(v, 1)
	f.elo, f.ehi = new(big.Rat), new(big.Rat)
	return f
}

func fsample(i int) form {
	f := fconst(0)
	f.c[i] = big.NewRat(1, 1)
	return f
}

func (a form) lin(ka int64, b form, kb int64) form {
	var f form
	ra, rb := big.NewRat(ka, 1), big.NewRat(kb, 1)
	mul := func(x, k *big.Rat) *big.Rat { return new(big.Rat).Mul(x, k) }
	for i := range f.c {
		f.c[i] = new(big.Rat).Add(mul(a.c[i], ra), mul(b.c[i], rb))
	}
	f.c0 = new(big.Rat).Add(mul(a.c0, ra), mul(b.c0, rb))
	lo := new(big.Rat)
	hi := new(big.Rat)
	for _, t := range []struct {
		lo, hi, k *big.Rat
	}{{a.elo, a.ehi, ra}, {b.elo, b.ehi, rb}} {
		x, y := mul(t.lo, t.k), mul(t.hi, t.k)
		if x.Cmp(y) > 0 {
			x, y = y, x
		}
		lo.Add(lo, x)
		hi.Add(hi, y)
	}
	f.elo, f.ehi = lo, hi
	return f
}

func (a form) add(b form) form  { return a.lin(1, b, 1) }
func (a form) sub(b form) form  { return a.lin(1, b, -1) }
func (a form) mul(k int32) form { return a.lin(int64(k), fconst(0), 0) }
func (a form) shl(n uint) form  { return a.lin(1<<n, fconst(0), 0) }
func (a form) sshr(n uint) form {
	d := big.NewRat(1, 1<<n)
	var f form
	for i := range f.c {
		f.c[i] = new(big.Rat).Mul(a.c[i], d)
	}
	f.c0 = new(big.Rat).Mul(a.c0, d)
	f.elo = new(big.Rat).Sub(new(big.Rat).Mul(a.elo, d), new(big.Rat).Sub(big.NewRat(1, 1), d))
	f.ehi = new(big.Rat).Mul(a.ehi, d)
	return f
}

// bounds returns the range of a over all sample blocks.
func (a form) bounds() (lo, hi *big.Rat) {
	lo = new(big.Rat).Add(a.c0, a.elo)
	hi = new(big.Rat).Add(a.c0, a.ehi)
	k := big.NewRat(255, 1)
	for _, c := range a.c {
		v := new(big.Rat).Mul(c, k)
		if v.Sign() < 0 {
			lo.Add(lo, v)
		} else {
			hi.Add(hi, v)
		}
	}
	return lo, hi
}

// eval returns the interval a gives for the samples s.
func (a form) eval(s *[BlockSize]byte) (lo, hi *big.Rat) {
	v := new(big.Rat).Set(a.c0)
	for i, c := range a.c {
		v.Add(v, new(big.Rat).Mul(c, big.NewRat(int64(s[i]), 1)))
	}
	return new(big.Rat).Add(v, a.elo), new(big.Rat).Add(v, a.ehi)
}

// fdctForms runs fdctCols and fdctRows of dct.go on forms, checking that
// every intermediate value fits in int32, and returns the fdctRows outputs
// before the final (x + 1<<17) >> 18.
func fdctForms(t *testing.T, b *fblock) fblock {
	chk := func(f form) form {
		lo, hi := f.bounds()
		if lo.Cmp(big.NewRat(math.MinInt32, 1)) < 0 || hi.Cmp(big.NewRat(math.MaxInt32, 1)) > 0 {
			t.Fatalf("intermediate value [%s, %s] outside int32", lo.FloatString(1), hi.FloatString(1))
		}
		return f
	}
	box := func(x0, x1 form, kcos, ksin int32) (form, form) {
		ksum := chk(chk(x0.add(x1)).mul(kcos))
		y0 := chk(ksum.add(chk(x1.mul(ksin - kcos))))
		y1 := chk(ksum.sub(chk(x0.mul(kcos + ksin))))
		return y0, y1
	}
	bfly := func(x, y form) (form, form) { return chk(x.add(y)), chk(x.sub(y)) }

	// fdctCols.
	for i := 0; i < 8; i++ {
		x0, x1, x2, x3 := b[0*8+i], b[1*8+i], b[2*8+i], b[3*8+i]
		x4, x5, x6, x7 := b[4*8+i], b[5*8+i], b[6*8+i], b[7*8+i]
		x0, x7 = bfly(x0, x7)
		x1, x6 = bfly(x1, x6)
		x2, x5 = bfly(x2, x5)
		x3, x4 = bfly(x3, x4)
		x4, x7 = box(x4, x7, c(cos3, 18), c(sin3, 18))
		x5, x6 = box(x5, x6, c(cos1, 18), c(sin1, 18))
		x0, x3 = bfly(x0, x3)
		x1, x2 = bfly(x1, x2)
		x2, x3 = box(x2, x3, c(sqrt2_cos6, 18), c(sqrt2_sin6, 18))
		x0, x1 = bfly(x0, x1)
		b[0*8+i] = chk(chk(x0.sub(fconst(128 * 8))).shl(18))
		b[4*8+i] = chk(x1.shl(18))
		b[2*8+i] = x2
		b[6*8+i] = x3
		x4, x6 = bfly(x4, x6)
		x7, x5 = bfly(x7, x5)
		x5 = chk(x5.sshr(12).mul(c(sqrt2, 12)))
		x6 = chk(x6.sshr(12).mul(c(sqrt2, 12)))
		x7, x4 = bfly(x7, x4)
		b[1*8+i] = x7
		b[3*8+i] = x5
		b[5*8+i] = x6
		b[7*8+i] = x4
	}
	// fdctRows, without the final cut.
	var out fblock
	for i := 0; i < 8; i++ {
		x := b[8*i : 8*i+8]
		x0, x1, x2, x3, x4, x5, x6, x7 := x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
		x0, x7 = bfly(x0, x7)
		x1, x6 = bfly(x1, x6)
		x2, x5 = bfly(x2, x5)
		x3, x4 = bfly(x3, x4)
		x4, x7 = box(x4.sshr(14), x7.sshr(14), c(cos3, 14), c(sin3, 14))
		x5, x6 = box(x5.sshr(14), x6.sshr(14), c(cos1, 14), c(sin1, 14))
		x0, x3 = bfly(x0, x3)
		x1, x2 = bfly(x1, x2)
		x2, x3 = box(x2.sshr(14), x3.sshr(14), c(sqrt2_cos6, 14), c(sqrt2_sin6, 14))
		x0, x1 = bfly(x0, x1)
		x4, x6 = bfly(x4, x6)
		x7, x5 = bfly(x7, x5)
		x5 = chk(x5.sshr(14).mul(c(sqrt2, 14)))
		x6 = chk(x6.sshr(14).mul(c(sqrt2, 14)))
		x7, x4 = bfly(x7, x4)
		o := out[8*i : 8*i+8]
		o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7] = x0, x7, x2, x5, x1, x6, x3, x4
	}
	return out
}
