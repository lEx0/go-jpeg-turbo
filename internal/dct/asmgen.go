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

//go:build ignore

// asmgen writes dct_arm64.s, the NEON kernels of package dct:
//
//	go run asmgen.go [-o file]
//
// The kernels are straight-line translations of dct.go (the Go 1.27
// image/jpeg DCT) into 32-bit vector lanes: every Go int32 product, sum,
// difference and shift becomes the same NEON operation on four lanes (MUL,
// MLA, MLS, ADD, SUB, SHL and SSHR wrap and round exactly like Go), so the
// results are bit-exact for every input, including inputs whose int32
// arithmetic overflows. The only rewrites are identities that hold in the
// ring of int32 values with wrapping arithmetic (Z/2^32), listed where they
// are used:
//
//   - dctBox(x0, x1, kcos, ksin) computes ksum = kcos*(x0+x1),
//     y0 = ksum + (ksin-kcos)*x1 and y1 = ksum - (kcos+ksin)*x0, which are
//     kcos*x0 + ksin*x1 and kcos*x1 - ksin*x0 in Z/2^32: MUL + MLA and
//     MUL + MLS by element (box).
//   - ADDHN/SUBHN (the high half of a wrapping sum or difference) fuse a
//     final butterfly with the v>>16 narrowing, which is exact because v>>16
//     of an int32 fits in 16 bits.
//   - Adding a constant anywhere along a chain of additions gives the same
//     int32 result (the rounding term of idctCols, see idctDoc).
//
// A generator is used instead of macros because the passes are written once
// and instantiated on different registers: it does the register allocation
// (lowest free register first) and emits the instructions the Go assembler
// lacks as WORD macros, whose encodings were checked against clang.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
)

// reg is a NEON register number (V0-V31).
type reg int

// none asks a helper to choose the destination register itself.
const none reg = -1

func (r reg) String() string { return fmt.Sprintf("V%d", int(r)) }

type gen struct {
	b    bytes.Buffer
	used [32]bool
	peak int
}

func (g *gen) emit(format string, args ...any) {
	fmt.Fprintf(&g.b, "\t"+format+"\n", args...)
}

func (g *gen) text(s string) { g.b.WriteString(s) }

func (g *gen) note() {
	n := 0
	for _, u := range g.used {
		if u {
			n++
		}
	}
	g.peak = max(g.peak, n)
}

// alloc returns the lowest free register.
func (g *gen) alloc() reg {
	for i, u := range g.used {
		if !u {
			g.used[i] = true
			g.note()
			return reg(i)
		}
	}
	panic("out of vector registers")
}

// allocRun returns the first register of the lowest run of n free
// consecutive registers (for register lists of LD1/ST1/LD4/TBL).
func (g *gen) allocRun(n int) reg {
	for i := 0; i+n <= len(g.used); i++ {
		ok := true
		for j := i; j < i+n; j++ {
			ok = ok && !g.used[j]
		}
		if ok {
			for j := i; j < i+n; j++ {
				g.used[j] = true
			}
			g.note()
			return reg(i)
		}
	}
	panic(fmt.Sprintf("no run of %d free vector registers", n))
}

func (g *gen) free(rs ...reg) {
	for _, r := range rs {
		if !g.used[r] {
			panic(fmt.Sprintf("%v freed twice", r))
		}
		g.used[r] = false
	}
}

func (g *gen) freeRun(r reg, n int) {
	for i := 0; i < n; i++ {
		g.free(r + reg(i))
	}
}

func (g *gen) allFree() {
	for i, u := range g.used {
		if u {
			panic(fmt.Sprintf("V%d still allocated", i))
		}
	}
}

func list(r reg, n int, t string) string {
	var s []string
	for i := 0; i < n; i++ {
		s = append(s, fmt.Sprintf("%v.%s", r+reg(i), t))
	}
	return "[" + strings.Join(s, ", ") + "]"
}

// --- instruction helpers -----------------------------------------------------

// op3 emits d = n OP m (Go operand order: OP Vm, Vn, Vd).
func (g *gen) op3(op, t string, d, n, m reg) {
	g.emit("%s\t%v.%s, %v.%s, %v.%s", op, m, t, n, t, d, t)
}

// imm emits d = OP(n, #k).
func (g *gen) imm(op string, k int, tn string, n reg, td string, d reg) {
	g.emit("%s\t$%d, %v.%s, %v.%s", op, k, n, tn, d, td)
}

// word emits a WORD macro m(d, n, x) with the GNU syntax as comment.
func (g *gen) word(m string, gnu string, args ...reg) {
	var s []string
	for _, a := range args {
		s = append(s, fmt.Sprint(int(a)))
	}
	g.emit("%s(%s) // %s", m, strings.Join(s, ", "), gnu)
}

func (g *gen) add(d, n, m reg)      { g.op3("VADD", "S4", d, n, m) }
func (g *gen) sub(d, n, m reg)      { g.op3("VSUB", "S4", d, n, m) }
func (g *gen) shl(d, n reg, k int)  { g.imm("VSHL", k, "S4", n, "S4", d) }
func (g *gen) sshr(d, n reg, k int) { g.imm("VSSHR", k, "S4", n, "S4", d) }
func (g *gen) mul(d, n, m reg)      { g.op3("VMUL", "S4", d, n, m) }

// mule, mlae, mlse: d = n*m.S[i], d += n*m.S[i], d -= n*m.S[i].
func (g *gen) mule(d, n, m reg, i int) {
	g.emit("MULE(%d, %d, %d, %d) // mul %v.4s, %v.4s, %v.s[%d]", d, n, m, i, d, n, m, i)
}

func (g *gen) mlae(d, n, m reg, i int) {
	g.emit("MLAE(%d, %d, %d, %d) // mla %v.4s, %v.4s, %v.s[%d]", d, n, m, i, d, n, m, i)
}

func (g *gen) mlse(d, n, m reg, i int) {
	g.emit("MLSE(%d, %d, %d, %d) // mls %v.4s, %v.4s, %v.s[%d]", d, n, m, i, d, n, m, i)
}

// bflyTo is the butterfly s, d = x+y, x-y on 32-bit lanes. A destination
// given as none is allocated (s) or is y itself (d).
func (g *gen) bflyTo(x, y, s, d reg) (reg, reg) {
	if s == none {
		s = g.alloc()
	}
	g.add(s, x, y)
	if d == none {
		d = y
	}
	g.sub(d, x, y)
	g.free(x)
	if d != y {
		g.free(y)
	}
	return s, d
}

func (g *gen) bfly(x, y reg) (reg, reg) { return g.bflyTo(x, y, none, none) }

// bfly16 is bfly on 16-bit lanes.
func (g *gen) bfly16(x, y reg) (reg, reg) {
	s := g.alloc()
	g.op3("VADD", "H8", s, x, y)
	g.op3("VSUB", "H8", y, x, y)
	g.free(x)
	return s, y
}

// box is dctBox(x0, x1, kcos, ksin) with kcos = k.S[lc], ksin = k.S[ls]:
// y0 = kcos*x0 + ksin*x1, y1 = kcos*x1 - ksin*x0 (in Z/2^32, the values of
// dct.go's ksum + (ksin-kcos)*x1 and ksum - (kcos+ksin)*x0). A destination
// given as none is allocated (y0) or is x1 itself (y1).
func (g *gen) box(x0, x1, k reg, lc, ls int, y0, y1 reg) (reg, reg) {
	if y0 == none {
		y0 = g.alloc()
	}
	g.mule(y0, x0, k, lc)
	g.mlae(y0, x1, k, ls)
	if y1 == none {
		y1 = x1
	}
	g.mule(y1, x1, k, lc)
	g.mlse(y1, x0, k, ls)
	g.free(x0)
	if y1 != x1 {
		g.free(x1)
	}
	return y0, y1
}

// transpose4 transposes the 4x4 int32 matrix held in r (one row per
// register) and returns its columns.
func (g *gen) transpose4(r [4]reg) [4]reg {
	t0, t1 := g.alloc(), g.alloc()
	g.op3("VTRN1", "S4", t0, r[0], r[1])
	g.op3("VTRN2", "S4", t1, r[0], r[1])
	g.free(r[0], r[1])
	t2, t3 := g.alloc(), g.alloc()
	g.op3("VTRN1", "S4", t2, r[2], r[3])
	g.op3("VTRN2", "S4", t3, r[2], r[3])
	g.free(r[2], r[3])
	var c [4]reg
	c[0], c[2] = g.alloc(), g.alloc()
	g.op3("VTRN1", "D2", c[0], t0, t2)
	g.op3("VTRN2", "D2", c[2], t0, t2)
	g.free(t0, t2)
	c[1], c[3] = g.alloc(), g.alloc()
	g.op3("VTRN1", "D2", c[1], t1, t3)
	g.op3("VTRN2", "D2", c[3], t1, t3)
	g.free(t1, t3)
	return c
}

// orReduce returns a new register holding the OR of rs; rs[i] is freed if
// owned[i].
func (g *gen) orReduce(rs []reg, owned []bool) reg {
	type v struct {
		r     reg
		owned bool
	}
	vs := make([]v, len(rs))
	for i, r := range rs {
		vs[i] = v{r, owned[i]}
	}
	for len(vs) > 1 {
		var next []v
		for i := 0; i+1 < len(vs); i += 2 {
			a, b := vs[i], vs[i+1]
			d := g.alloc()
			g.op3("VORR", "B16", d, a.r, b.r)
			if a.owned {
				g.free(a.r)
			}
			if b.owned {
				g.free(b.r)
			}
			next = append(next, v{d, true})
		}
		if len(vs)%2 == 1 {
			next = append(next, vs[len(vs)-1])
		}
		vs = next
	}
	if !vs[0].owned {
		panic("orReduce of a single register")
	}
	return vs[0].r
}

// --- constants ------------------------------------------------------------------

// The 60-bit constants of dct.go and its rounding helper c.
const (
	cos1          = 1130768441178740757
	sin1          = 224923827593068887
	cos3          = 958619196450722178
	sin3          = 640528868967736374
	sqrt2         = 1630477228166597777
	sqrt2_cos6    = 623956622067911264
	sqrt2_sin6    = 1506364539328854985
	sqrt2inv      = 815238614083298888
	sqrt2inv_cos6 = 311978311033955632
	sqrt2inv_sin6 = 753182269664427492
)

func c(x uint64, bits int) int32 {
	return int32((x + (1 << (59 - bits))) >> (60 - bits))
}

type vec struct {
	v    [4]int32
	note string
}

func (g *gen) data32(name string, vs ...vec) {
	off := 0
	for _, v := range vs {
		for i := 0; i < 4; i += 2 {
			lo, hi := uint32(v.v[i]), uint32(v.v[i+1])
			cm := ""
			if i == 0 {
				cm = " // " + v.note
			}
			g.text(fmt.Sprintf("DATA %s<>+%d(SB)/8, $0x%08X%08X%s\n", name, off, hi, lo, cm))
			off += 8
		}
	}
	g.text(fmt.Sprintf("GLOBL %s<>(SB), RODATA|NOPTR, $%d\n\n", name, off))
}

func (g *gen) data8(name string, comment []string, rows ...[16]byte) {
	off := 0
	for k, v := range rows {
		for i := 0; i < 16; i += 8 {
			var x uint64
			for j := 7; j >= 0; j-- {
				x = x<<8 | uint64(v[i+j])
			}
			cm := ""
			if i == 0 && k < len(comment) {
				cm = " // " + comment[k]
			}
			g.text(fmt.Sprintf("DATA %s<>+%d(SB)/8, $0x%016X%s\n", name, off, x, cm))
			off += 8
		}
	}
	g.text(fmt.Sprintf("GLOBL %s<>(SB), RODATA|NOPTR, $%d\n\n", name, off))
}

func splat16(v int32) [4]int32 {
	x := v&0xFFFF | v<<16
	return [4]int32{x, x, x, x}
}

func vecs(vs ...[4]int32) []vec {
	r := make([]vec, len(vs))
	for i, v := range vs {
		r[i] = vec{v: v}
	}
	return r
}

// idctConst (registers kA, kB, kC, kR of genIDCT).
var idctConst = []vec{
	{[4]int32{c(sqrt2inv_cos6, 18), -c(sqrt2inv_sin6, 18), c(sqrt2inv, 8), c(sqrt2inv, 14)},
		"kA: rows box(x2, x3) kcos, ksin; c(sqrt2inv, 8); c(sqrt2inv, 14)"},
	{[4]int32{c(cos3, 12), -c(sin3, 12), c(cos1, 12), -c(sin1, 12)},
		"kB: box(x4, x7) kcos, ksin; box(x5, x6) kcos, ksin"},
	{[4]int32{c(sqrt2inv_cos6, 12), -c(sqrt2inv_sin6, 12), c(sqrt2inv_sin6, 12), 0},
		"kC: cols box(x2, x3) kcos, ksin, -ksin"},
	{[4]int32{4, 0, 0, 0}, "kR: rounding term of row 0 (4<<17 = 1<<19)"},
}

// fdctConst (registers kA, kC, k1024, then kD, kE of genFDCT).
var fdctConst = []vec{
	{[4]int32{c(cos3, 18), c(sin3, 18), c(cos1, 18), c(sin1, 18)},
		"kA: cols box(x4, x7) kcos, ksin; box(x5, x6) kcos, ksin"},
	{[4]int32{c(sqrt2_cos6, 18), c(sqrt2_sin6, 18), c(sqrt2, 12), 0},
		"kC: cols box(x2, x3) kcos, ksin; c(sqrt2, 12)"},
	{splat16(1024), "k1024: 1024 in eight int16 lanes"},
	{[4]int32{c(cos3, 14), c(sin3, 14), c(cos1, 14), c(sin1, 14)},
		"kD: rows box(x4, x7) kcos, ksin; box(x5, x6) kcos, ksin"},
	{[4]int32{c(sqrt2_cos6, 14), c(sqrt2_sin6, 14), c(sqrt2, 14), 0},
		"kE: rows box(x2, x3) kcos, ksin; c(sqrt2, 14)"},
}

var unzig = [64]int{
	0, 1, 8, 16, 9, 2, 3, 10,
	17, 24, 32, 25, 18, 11, 4, 5,
	12, 19, 26, 33, 40, 48, 41, 34,
	27, 20, 13, 6, 7, 14, 21, 28,
	35, 42, 49, 56, 57, 50, 43, 36,
	29, 22, 15, 23, 30, 37, 44, 51,
	58, 59, 52, 45, 38, 31, 39, 46,
	53, 60, 61, 54, 47, 55, 62, 63,
}

// --- inverse DCT ------------------------------------------------------------------

type idctRegs struct{ a, b, c, r reg }

// idctRowPass is idctRows on four rows held as eight column vectors (lane i
// of in[k] is coefficient k of row i). The output columns 0-3 go to the run
// lo..lo+3 and columns 4-7 to hi..hi+3. With round, lane 0 (row 0) also
// gets idctCols' rounding term (see idctDoc).
func (g *gen) idctRowPass(in [8]reg, k idctRegs, round bool, lo, hi reg) {
	x0, x7, x2, x5, x1, x6, x3, x4 := in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7]
	if round {
		g.add(x0, x0, k.r)
	}
	g.shl(x0, x0, 17)
	g.shl(x1, x1, 17)
	x0, x1 = g.bfly(x0, x1)
	x2, x3 = g.box(x2, x3, k.a, 0, 1, none, none)
	x1, x2 = g.bfly(x1, x2)
	x0, x3 = g.bfly(x0, x3)

	g.shl(x4, x4, 7)
	g.shl(x7, x7, 7)
	x7, x4 = g.bfly(x7, x4)
	g.mule(x6, x6, k.a, 2)
	g.mule(x5, x5, k.a, 2)
	x7, x5 = g.bfly(x7, x5)
	x4, x6 = g.bfly(x4, x6)
	g.sshr(x4, x4, 2)
	g.sshr(x7, x7, 2)
	x4, x7 = g.box(x4, x7, k.b, 0, 1, none, none)
	g.sshr(x5, x5, 2)
	g.sshr(x6, x6, 2)
	x5, x6 = g.box(x5, x6, k.b, 2, 3, none, none)

	g.bflyTo(x0, x7, lo, hi+3)
	g.bflyTo(x1, x6, lo+1, hi+2)
	g.bflyTo(x2, x5, lo+2, hi+1)
	g.bflyTo(x3, x4, lo+3, hi)
}

// idctColPass is idctCols up to (excluding) the final butterflies, on four
// columns held as eight row vectors (lane j of in[k] is row k of column j).
// It returns x0..x7; the outputs are rows y = x_y + x_(7-y) for y < 4 and
// x_(7-y) - x_y for y >= 4, then >>18. The rounding term 1<<19 has already
// been added to in[0] by the row pass.
func (g *gen) idctColPass(in [8]reg, k idctRegs) [8]reg {
	x0, x7, x2, x5, x1, x6, x3, x4 := in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7]
	x0, x1 = g.bfly(x0, x1)
	g.sshr(x0, x0, 2)
	g.sshr(x1, x1, 2)
	g.sshr(x2, x2, 13)
	g.sshr(x3, x3, 13)
	x2, x3 = g.box(x2, x3, k.c, 0, 1, none, none)
	x1, x2 = g.bfly(x1, x2)
	x0, x3 = g.bfly(x0, x3)

	x7, x4 = g.bfly(x7, x4)
	g.sshr(x5, x5, 13)
	g.mule(x5, x5, k.a, 3)
	g.sshr(x6, x6, 13)
	g.mule(x6, x6, k.a, 3)
	x7, x5 = g.bfly(x7, x5)
	x4, x6 = g.bfly(x4, x6)
	g.sshr(x4, x4, 14)
	g.sshr(x7, x7, 14)
	x4, x7 = g.box(x4, x7, k.b, 0, 1, none, none)
	g.sshr(x5, x5, 14)
	g.sshr(x6, x6, 14)
	x5, x6 = g.box(x5, x6, k.b, 2, 3, none, none)
	return [8]reg{x0, x1, x2, x3, x4, x5, x6, x7}
}

// idctColPassTop is idctColPass for rows 4-7 zero (in holds rows 0-3):
// idctCols with x1 = x6 = x3 = x4 = 0, simplified by
//
//	x0, x1 = (x0+0)>>2, (x0-0)>>2           -> both A = x0>>2
//	dctBox(u, 0) with u = x2>>13            -> kcos*u, -ksin*u
//	x7, x4 = x7+0, x7-0; x6 = (0>>13)*c = 0 -> x4 = x6 = x7 (X7) after
//	                                           x4, x6 = x4+x6, x4-x6
//
// so both boxes read p = X7>>14. All of these are int32 identities (Z/2^32).
func (g *gen) idctColPassTop(in [4]reg, k idctRegs) [8]reg {
	a, x7, u, x5 := in[0], in[1], in[2], in[3]
	g.sshr(a, a, 2)
	g.sshr(u, u, 13)
	y0 := g.alloc()
	g.mule(y0, u, k.c, 0) // kcos*u
	g.mule(u, u, k.c, 2)  // -ksin*u
	y1 := u
	x1 := g.alloc()
	g.add(x1, a, y0)
	g.sub(y0, a, y0) // x2
	x0 := g.alloc()
	g.add(x0, a, y1)
	g.sub(y1, a, y1) // x3
	g.free(a)
	x2, x3 := y0, y1

	g.sshr(x5, x5, 13)
	g.mule(x5, x5, k.a, 3)
	p := g.alloc()
	g.sshr(p, x7, 14)
	x7, x5 = g.bfly(x7, x5)
	g.sshr(x7, x7, 14) // q
	g.sshr(x5, x5, 14) // r
	// x4, x7 = dctBox(p, q): kcos*p + ksin*q, kcos*q - ksin*p.
	x4 := g.alloc()
	g.mule(x4, p, k.b, 0)
	g.mlae(x4, x7, k.b, 1)
	g.mule(x7, x7, k.b, 0)
	g.mlse(x7, p, k.b, 1)
	// x5, x6 = dctBox(r, p): kcos*r + ksin*p, kcos*p - ksin*r.
	x5n := g.alloc()
	g.mule(x5n, x5, k.b, 2)
	g.mlae(x5n, p, k.b, 3)
	g.mule(p, p, k.b, 2)
	g.mlse(p, x5, k.b, 3)
	g.free(x5)
	return [8]reg{x0, x1, x2, x3, x4, x5n, p, x7}
}

// genIDCT emits dequantIDCTStoreNEON (store) or dequantIDCTNEON.
func (g *gen) genIDCT(store bool) {
	name, frame := "dequantIDCTNEON", "$0-16"
	if store {
		name, frame = "dequantIDCTStoreNEON", "$0-32"
	}
	g.text(fmt.Sprintf("TEXT ·%s(SB), NOSPLIT, %s\n", name, frame))
	g.emit("MOVD\tb+0(FP), R0")
	g.emit("MOVD\tqt+8(FP), R1")
	if store {
		g.emit("MOVD\tdst+16(FP), R2")
		g.emit("MOVD\tstride+24(FP), R3")
	}

	// b in V0-V15: LD4 puts b[16i + 4l + j] in lane l of V[4i+j], so V[4i+j]
	// holds rows 2i and 2i+1 of the columns j and j+4.
	bv := g.allocRun(16)
	g.emit("MOVD\tR0, R4")
	for i := 0; i < 4; i++ {
		post := ".P\t64(R4)"
		if i == 3 {
			post = "\t(R4)"
		}
		g.emit("VLD4%s, %s", post, list(bv+reg(4*i), 4, "S4"))
	}

	g.text("\n\t// DC only? (OR of the 63 AC coefficients: V0 without lane 0, V1-V15.)\n")
	g.text("\t// Rows 4-7 zero? (OR of V8-V15.)\n")
	t := g.alloc()
	g.emit("VEXT\t$4, %v.B16, %v.B16, %v.B16", bv+1, bv, t)
	rs, owned := []reg{t}, []bool{true}
	for i := 1; i < 8; i++ {
		rs, owned = append(rs, bv+reg(i)), append(owned, false)
	}
	top := g.orReduce(rs, owned)
	rs, owned = nil, nil
	for i := 8; i < 16; i++ {
		rs, owned = append(rs, bv+reg(i)), append(owned, false)
	}
	bot := g.orReduce(rs, owned)
	all := g.alloc()
	g.op3("VORR", "B16", all, top, bot)
	g.free(top)
	g.emit("VUMAXV\t%v.S4, %v", all, all)
	g.emit("VUMAXV\t%v.S4, %v", bot, bot)
	g.emit("VMOV\t%v.S[0], R6", all)
	g.emit("VMOV\t%v.S[0], R7", bot)
	g.free(all, bot)
	g.emit("CBZW\tR6, dconly")
	g.emit("CBZW\tR7, sparse")

	saved := g.used
	g.idctBody(store, bv, false)
	g.used = saved
	g.text("\n// Rows 4-7 of the coefficients are zero, so rows 4-7 of idctRows' output\n")
	g.text("// are zero too (every term is a product, shift or box of zeros): skip\n")
	g.text("// their dequantization and row pass, and simplify idctCols for them\n")
	g.text("// (idctColPassTop in asmgen.go).\n")
	g.text("sparse:\n")
	g.idctBody(store, bv, true)

	g.text("\n// DC only: every output is ((b[0]*qt[0])<<17 + 1<<19) >> 20 (int32,\n")
	g.text("// wrapping; see dcValue in kernel_test.go).\n")
	g.text("dconly:\n")
	g.emit("MOVW\t(R0), R6")
	g.emit("MOVW\t(R1), R7")
	g.emit("MULW\tR7, R6, R6")
	g.emit("LSLW\t$17, R6, R6")
	g.emit("ADDW\t$(1<<19), R6, R6")
	g.emit("ASRW\t$20, R6, R6")
	if store {
		g.emit("ADDW\t$128, R6, R6")
		g.emit("VDUP\tR6, V0.H8")
		g.emit("VSQXTUN\tV0.H8, V0.B8")
		for r := 0; r < 7; r++ {
			g.emit("VST1.P\tV0.D[0], (R2)(R3)")
		}
		g.emit("VST1\tV0.D[0], (R2)")
	} else {
		g.emit("VDUP\tR6, V0.S4")
		g.emit("VDUP\tR6, V1.S4")
		for r := 0; r < 8; r++ {
			g.emit("FSTPQ\t(F0, F1), %d(R0)", 32*r)
		}
	}
	g.emit("RET")
	g.allFree()
	g.text("\n")
}

// idctBody emits the inverse DCT after the checks, on the coefficients
// loaded in bv..bv+15, up to its RET. With sparse, rows 4-7 are zero.
func (g *gen) idctBody(store bool, bv reg, sparse bool) {
	groups := 4
	if sparse {
		groups = 2
		g.freeRun(bv+8, 8)
	}
	g.text("\n\t// Constants and dequantization (qt in the same LD4 order).\n")
	kc := g.allocRun(4)
	k := idctRegs{a: kc, b: kc + 1, c: kc + 2, r: kc + 3}
	g.emit("MOVD\t$idctConst<>(SB), R5")
	g.emit("VLD1\t(R5), %s", list(kc, 4, "S4"))
	g.emit("MOVD\tR1, R6")
	q := g.allocRun(4)
	for i := 0; i < groups; i++ {
		post := ".P\t64(R6)"
		if i == groups-1 {
			post = "\t(R6)"
		}
		g.emit("VLD4%s, %s", post, list(q, 4, "S4"))
		for j := 0; j < 4; j++ {
			r := bv + reg(4*i+j)
			g.mul(r, r, q+reg(j))
		}
	}
	g.freeRun(q, 4)

	g.text("\n\t// UZP1/UZP2 separate the columns: h[0][c] = column c of rows 0-3, h[1][c] of rows 4-7.\n")
	var h [2][8]reg
	for j := 0; j < 4; j++ {
		for half := 0; half < groups/2; half++ {
			a, b := bv+reg(8*half+j), bv+reg(8*half+4+j)
			h[half][j] = g.alloc()
			g.op3("VUZP1", "S4", h[half][j], a, b)
			h[half][j+4] = g.alloc()
			g.op3("VUZP2", "S4", h[half][j+4], a, b)
			g.free(a, b)
		}
	}

	// The row pass outputs (column vectors, lane = row) are transposed per
	// 4x4 quarter into the row vectors (lane = column) of idctCols.
	var o [2][8]reg
	for half := 0; half < groups/2; half++ {
		g.text(fmt.Sprintf("\n\t// idctRows, rows %d-%d.\n", 4*half, 4*half+3))
		lo, hi := g.allocRun(4), g.allocRun(4)
		g.idctRowPass(h[half], k, half == 0, lo, hi)
		for i := 0; i < 4; i++ {
			o[half][i], o[half][4+i] = lo+reg(i), hi+reg(i)
		}
	}
	g.free(k.r)
	var y [2][8]reg // [column half][x]
	for half := 0; half < 2; half++ {
		g.text(fmt.Sprintf("\n\t// Transpose, then idctCols, columns %d-%d.\n", 4*half, 4*half+3))
		top := g.transpose4([4]reg(o[0][4*half : 4*half+4]))
		if sparse {
			y[half] = g.idctColPassTop(top, k)
			continue
		}
		var in [8]reg
		bot := g.transpose4([4]reg(o[1][4*half : 4*half+4]))
		copy(in[0:4], top[:])
		copy(in[4:8], bot[:])
		y[half] = g.idctColPass(in, k)
	}
	g.free(k.a, k.b, k.c)

	if store {
		// Row r is (x_r + x_(7-r)) >> 18 (r < 4) or (x_(7-r) - x_r) >> 18,
		// clamped after adding 128. ADDHN/SUBHN give the wrapping sum or
		// difference >> 16 exactly (it fits in 16 bits), SQSHRN #2 clamps
		// h>>2 to [-128, 127] and the EOR with 0x80 adds 128.
		g.text("\n\t// Final butterflies, >>16, then clamp((v>>18) + 128) to bytes.\n")
		var hs [8]reg
		for p := 0; p < 4; p++ {
			for _, sum := range []bool{true, false} {
				r := p
				m, gnu := "ADDHN", "addhn"
				if !sum {
					r = 7 - p
					m, gnu = "SUBHN", "subhn"
				}
				d := g.alloc()
				a, b := y[0][p], y[0][7-p]
				g.word(m, fmt.Sprintf("%s %v.4h, %v.4s, %v.4s", gnu, d, a, b), d, a, b)
				a, b = y[1][p], y[1][7-p]
				g.word(m+"2", fmt.Sprintf("%s2 %v.8h, %v.4s, %v.4s", gnu, d, a, b), d, a, b)
				hs[r] = d
			}
			g.free(y[0][p], y[0][7-p], y[1][p], y[1][7-p])
		}
		c80 := g.alloc()
		g.emit("VMOVI\t$0x80, %v.B16", c80)
		for r := 0; r < 8; r += 2 {
			w := hs[r]
			g.word("SQSHRN8B", fmt.Sprintf("sqshrn %v.8b, %v.8h, #2", w, w), w, w)
			g.word("SQSHRN16B", fmt.Sprintf("sqshrn2 %v.16b, %v.8h, #2", w, hs[r+1]), w, hs[r+1])
			g.op3("VEOR", "B16", w, w, c80)
			g.emit("VST1.P\t%v.D[0], (R2)(R3)", w)
			if r < 6 {
				g.emit("VST1.P\t%v.D[1], (R2)(R3)", w)
			} else {
				g.emit("VST1\t%v.D[1], (R2)", w)
			}
			g.free(hs[r], hs[r+1])
		}
		g.free(c80)
	} else {
		g.text("\n\t// Final butterflies and >>18 into b (natural order).\n")
		for p := 0; p < 4; p++ {
			var s, d [2]reg
			for half := 0; half < 2; half++ {
				s[half], d[half] = g.bfly(y[half][p], y[half][7-p])
				g.sshr(s[half], s[half], 18)
				g.sshr(d[half], d[half], 18)
			}
			g.emit("FSTPQ\t(F%d, F%d), %d(R0)", s[0], s[1], 32*p)
			g.emit("FSTPQ\t(F%d, F%d), %d(R0)", d[0], d[1], 32*(7-p))
			g.free(s[0], s[1], d[0], d[1])
		}
	}
	g.emit("RET")
	g.allFree()
}

// --- forward DCT and quantization ------------------------------------------------

// fdctRow is the result of fdctRowPass: the natural output columns 2, 3,
// 5, 6 (x2, x5, x6, x3) and the operands of the deferred final butterflies
// (column 0 = x0+x1, 4 = x0-x1, 1 = x7+x4, 7 = x7-x4).
type fdctRow struct{ x0, x1, x7, x4, c2, c3, c5, c6 reg }

// fdctRowPass is fdctRows without its last two butterflies and the final
// rounding shift, on four rows held as eight column vectors.
func (g *gen) fdctRowPass(in [8]reg, kd, ke reg) fdctRow {
	x0, x1, x2, x3, x4, x5, x6, x7 := in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7]
	x0, x7 = g.bfly(x0, x7)
	x1, x6 = g.bfly(x1, x6)
	x2, x5 = g.bfly(x2, x5)
	x3, x4 = g.bfly(x3, x4)

	g.sshr(x4, x4, 14)
	g.sshr(x7, x7, 14)
	x4, x7 = g.box(x4, x7, kd, 0, 1, none, none)
	g.sshr(x5, x5, 14)
	g.sshr(x6, x6, 14)
	x5, x6 = g.box(x5, x6, kd, 2, 3, none, none)
	x0, x3 = g.bfly(x0, x3)
	x1, x2 = g.bfly(x1, x2)

	g.sshr(x2, x2, 14)
	g.sshr(x3, x3, 14)
	x2, x3 = g.box(x2, x3, ke, 0, 1, none, none)
	x4, x6 = g.bfly(x4, x6)
	x7, x5 = g.bfly(x7, x5)

	g.sshr(x5, x5, 14)
	g.mule(x5, x5, ke, 2)
	g.sshr(x6, x6, 14)
	g.mule(x6, x6, ke, 2)
	return fdctRow{x0: x0, x1: x1, x7: x7, x4: x4, c2: x2, c3: x5, c5: x6, c6: x3}
}

// quantZig quantizes the eight column vectors n (int16 lane r of n[j] is the
// coefficient at row r, column j), reorders them into zig-zag order, stores
// them to (R3) and leaves the non-zero bitmap in R6. R2 points at
// Quant.vtab. n is freed.
func (g *gen) quantZig(n [8]reg) {
	g.text("\n\t// Quantize: q = sign(a) * ((|a| + bias) * mul >> (15+k)), see initVec.\n")
	zero := g.alloc()
	g.op3("VEOR", "B16", zero, zero, zero)
	// Two runs of four registers for the TBL tables.
	qa := g.allocRun(4)
	qb := g.allocRun(4)
	qr := [8]reg{qa, qa + 1, qa + 2, qa + 3, qb, qb + 1, qb + 2, qb + 3}
	for j := 0; j < 8; j++ {
		h := g.allocRun(3)
		g.emit("VLD1.P\t48(R2), %s // bias, mul, -k", list(h, 3, "H8"))
		g.word("SABA8H", fmt.Sprintf("saba %v.8h, %v.8h, %v.8h: x = bias + |a|", h, n[j], zero), h, n[j], zero)
		g.word("SQDMULH8H", fmt.Sprintf("sqdmulh %v.8h, %v.8h, %v.8h: (x*mul) >> 15", h, h, h+1), h, h, h+1)
		g.op3("VUSHL", "H8", h, h, h+2)
		g.imm("VSSHR", 15, "H8", n[j], "H8", h+1)
		g.op3("VEOR", "B16", h, h, h+1)
		g.op3("VSUB", "H8", qr[j], h, h+1)
		g.free(n[j], h, h+1, h+2)
	}
	g.free(zero)

	g.text("\n\t// Zig-zag order: TBL/TBX over {qa} (columns 0-3) and {qb} (columns 4-7).\n")
	g.emit("MOVD\t$fdctZig<>(SB), R5")
	z := g.allocRun(8)
	var pending []reg
	left := 0
	for m := 0; m < 8; m++ {
		left += len(zigGroups(m))
	}
	nextIdx := func() reg {
		if len(pending) == 0 {
			n := min(4, left)
			r := g.allocRun(n)
			for i := 0; i < n; i++ {
				pending = append(pending, r+reg(i))
			}
			g.emit("VLD1.P\t%d(R5), %s", 16*n, list(r, n, "B16"))
		}
		left--
		r := pending[0]
		pending = pending[1:]
		return r
	}
	for m := 0; m < 8; m++ {
		for i, grp := range zigGroups(m) {
			idx := nextIdx()
			tq := qa
			if grp == 1 {
				tq = qb
			}
			op := "VTBL"
			if i > 0 {
				op = "VTBX"
			}
			g.emit("%s\t%v.B16, %s, %v.B16", op, idx, list(tq, 4, "B16"), z+reg(m))
			g.free(idx)
		}
	}
	if len(pending) != 0 {
		panic("unused zig-zag index registers")
	}
	g.free(qr[:]...)

	g.text("\n\t// Bitmap: bit 63-zig set iff out[zig] != 0.\n")
	mask := g.alloc()
	g.emit("VLD1\t(R5), [%v.B16]", mask)
	var p [4]reg
	for i := range p {
		a, b := g.alloc(), g.alloc()
		g.op3("VCMTST", "H8", a, z+reg(2*i), z+reg(2*i))
		g.op3("VCMTST", "H8", b, z+reg(2*i+1), z+reg(2*i+1))
		g.op3("VUZP1", "B16", a, a, b)
		g.free(b)
		g.op3("VAND", "B16", a, a, mask)
		p[i] = a
	}
	g.free(mask)
	g.op3("VADDP", "B16", p[0], p[0], p[1])
	g.op3("VADDP", "B16", p[2], p[2], p[3])
	g.op3("VADDP", "B16", p[0], p[0], p[2])
	g.op3("VADDP", "B16", p[0], p[0], p[0])
	g.emit("VMOV\t%v.D[0], R6", p[0])
	g.emit("REV\tR6, R6")
	g.free(p[:]...)

	g.emit("VST1.P\t%s, 64(R3)", list(z, 4, "B16"))
	g.emit("VST1\t%s, (R3)", list(z+4, 4, "B16"))
	g.freeRun(z, 8)
}

// zigGroups returns the source groups (0: columns 0-3, 1: columns 4-7) of
// zig-zag output vector m, in TBL/TBX order.
func zigGroups(m int) []int {
	var has [2]bool
	for l := 0; l < 8; l++ {
		has[(unzig[8*m+l]%8)/4] = true
	}
	var gs []int
	for grp := 0; grp < 2; grp++ {
		if has[grp] {
			gs = append(gs, grp)
		}
	}
	return gs
}

// zigTables returns the TBL/TBX index vectors in the order quantZig uses
// them, and the bitmap mask.
func zigTables() ([][16]byte, []string) {
	var tabs [][16]byte
	var cms []string
	for m := 0; m < 8; m++ {
		for _, grp := range zigGroups(m) {
			var t [16]byte
			for l := 0; l < 8; l++ {
				n := unzig[8*m+l]
				r, col := n/8, n%8
				if col/4 != grp {
					t[2*l], t[2*l+1] = 0xFF, 0xFF
					continue
				}
				off := (col%4)*16 + 2*r
				t[2*l], t[2*l+1] = byte(off), byte(off+1)
			}
			tabs = append(tabs, t)
			cms = append(cms, fmt.Sprintf("zig %d-%d from columns %d-%d", 8*m, 8*m+7, 4*grp, 4*grp+3))
		}
	}
	tabs = append(tabs, [16]byte{0x80, 0x40, 0x20, 0x10, 8, 4, 2, 1, 0x80, 0x40, 0x20, 0x10, 8, 4, 2, 1})
	cms = append(cms, "bitmap mask")
	return tabs, cms
}

func (g *gen) genFDCT() {
	g.text("TEXT ·fdctQuantNEON(SB), NOSPLIT, $0-40\n")
	g.emit("MOVD\tsrc+0(FP), R0")
	g.emit("MOVD\tstride+8(FP), R1")
	g.emit("MOVD\tt+16(FP), R2")
	g.emit("MOVD\tout+24(FP), R3")
	g.emit("ADD\t$Quant_vtab, R2, R2")

	var r [8]reg
	for i := range r {
		r[i] = g.alloc()
		if i < 7 {
			g.emit("VLD1.P\t(R0)(R1), [%v.B8]", r[i])
		} else {
			g.emit("VLD1\t(R0), [%v.B8]", r[i])
		}
	}
	kc := g.allocRun(3)
	ka, kC, k1024 := kc, kc+1, kc+2
	g.emit("MOVD\t$fdctConst<>(SB), R5")
	g.emit("VLD1.P\t48(R5), %s", list(kc, 3, "S4"))

	g.text("\n\t// fdctCols stage 1, then the even part up to its box (16-bit lanes, all\n")
	g.text("\t// eight columns; every value is below 2^11).\n")
	var x [8]reg
	for i := 0; i < 4; i++ {
		x[i] = g.alloc()
		g.word("UADDL8H", fmt.Sprintf("uaddl %v.8h, %v.8b, %v.8b", x[i], r[i], r[7-i]), x[i], r[i], r[7-i])
		x[7-i] = g.alloc()
		g.word("USUBL8H", fmt.Sprintf("usubl %v.8h, %v.8b, %v.8b", x[7-i], r[i], r[7-i]), x[7-i], r[i], r[7-i])
		g.free(r[i], r[7-i])
	}
	x[0], x[3] = g.bfly16(x[0], x[3])
	x[1], x[2] = g.bfly16(x[1], x[2])
	x[0], x[1] = g.bfly16(x[0], x[1])
	g.op3("VSUB", "H8", x[0], x[0], k1024)
	g.free(k1024)

	// Each half's eight output rows (lane = column) are transposed as two
	// 4x4 quarters (lane = row).
	var h0, h1 [8]reg // h0[c] = column c of rows 0-3, h1[c] of rows 4-7
	for half := 0; half < 2; half++ {
		g.text(fmt.Sprintf("\n\t// fdctCols, columns %d-%d (32-bit lanes).\n", 4*half, 4*half+3))
		sx, sshll, th := "VSXTL", "VSSHLL", "H4"
		if half == 1 {
			sx, sshll, th = "VSXTL2", "VSSHLL2", "H8"
		}
		widen := func(v reg) reg {
			d := g.alloc()
			g.emit("%s\t%v.%s, %v.S4", sx, v, th, d)
			return d
		}
		top, bot := g.allocRun(4), g.allocRun(4) // rows 0-3, rows 4-7
		g.emit("%s\t$15, %v.%s, %v.S4", sshll, x[0], th, top)
		g.shl(top, top, 3)
		g.emit("%s\t$15, %v.%s, %v.S4", sshll, x[1], th, bot)
		g.shl(bot, bot, 3)
		x2, x3 := widen(x[2]), widen(x[3])
		g.box(x2, x3, kC, 0, 1, top+2, bot+2)
		x4, x5, x6, x7 := widen(x[4]), widen(x[5]), widen(x[6]), widen(x[7])
		x4, x7 = g.box(x4, x7, ka, 0, 1, none, none)
		x5, x6 = g.box(x5, x6, ka, 2, 3, none, none)
		x4, x6 = g.bfly(x4, x6)
		x7, x5 = g.bfly(x7, x5)
		g.sshr(x5, x5, 12)
		g.mule(top+3, x5, kC, 2)
		g.sshr(x6, x6, 12)
		g.mule(bot+1, x6, kC, 2)
		g.free(x5, x6)
		g.bflyTo(x7, x4, top+1, bot+3)

		a := g.transpose4([4]reg{top, top + 1, top + 2, top + 3})
		b := g.transpose4([4]reg{bot, bot + 1, bot + 2, bot + 3})
		for i := 0; i < 4; i++ {
			h0[4*half+i], h1[4*half+i] = a[i], b[i]
		}
	}
	g.free(x[:]...)
	g.free(ka, kC)

	kr := g.allocRun(2)
	kd, ke := kr, kr+1
	g.emit("VLD1\t(R5), %s", list(kr, 2, "S4"))
	g.text("\n\t// fdctRows, rows 0-3.\n")
	o0 := g.fdctRowPass(h0, kd, ke)
	g.text("\n\t// fdctRows, rows 4-7.\n")
	o1 := g.fdctRowPass(h1, kd, ke)
	g.free(kd, ke)

	// (v + 1<<17) >> 18 = ((v>>16) + 2) >> 2 = SRSHR(v>>16, #2), where v>>16
	// (exact in int16) is ADDHN/SUBHN of the last butterflies or UZP2 (the
	// high halves) of the other outputs.
	g.text("\n\t// Cut to Q13.0: n[c] = (v + 1<<17) >> 18 = SRSHR(v>>16, #2) in int16 lanes.\n")
	var n [8]reg
	hn := func(m string, col int, a0, b0, a1, b1 reg) {
		d := g.alloc()
		low := strings.ToLower(m)
		g.word(m, fmt.Sprintf("%s %v.4h, %v.4s, %v.4s", low, d, a0, b0), d, a0, b0)
		g.word(m+"2", fmt.Sprintf("%s2 %v.8h, %v.4s, %v.4s", low, d, a1, b1), d, a1, b1)
		n[col] = d
	}
	hn("ADDHN", 0, o0.x0, o0.x1, o1.x0, o1.x1)
	hn("SUBHN", 4, o0.x0, o0.x1, o1.x0, o1.x1)
	g.free(o0.x0, o0.x1, o1.x0, o1.x1)
	hn("ADDHN", 1, o0.x7, o0.x4, o1.x7, o1.x4)
	hn("SUBHN", 7, o0.x7, o0.x4, o1.x7, o1.x4)
	g.free(o0.x7, o0.x4, o1.x7, o1.x4)
	for _, cv := range []struct {
		col    int
		r0, r1 reg
	}{{2, o0.c2, o1.c2}, {3, o0.c3, o1.c3}, {5, o0.c5, o1.c5}, {6, o0.c6, o1.c6}} {
		d := g.alloc()
		g.op3("VUZP2", "H8", d, cv.r0, cv.r1)
		g.free(cv.r0, cv.r1)
		n[cv.col] = d
	}
	for j := range n {
		g.imm("VSRSHR", 2, "H8", n[j], "H8", n[j])
	}
	g.quantZig(n)
	g.emit("MOVD\tR6, ret+32(FP)")
	g.emit("RET")
	g.allFree()
	g.text("\n")
}

// genQuant emits quantNEON, the quantization part of fdctQuantNEON alone
// (for the exhaustive tests).
func (g *gen) genQuant() {
	g.text("TEXT ·quantNEON(SB), NOSPLIT, $0-32\n")
	g.emit("MOVD\tc+0(FP), R0")
	g.emit("MOVD\tt+8(FP), R2")
	g.emit("MOVD\tout+16(FP), R3")
	g.emit("ADD\t$Quant_vtab, R2, R2")
	var n [8]reg
	for i := 0; i < 8; i += 4 {
		r := g.allocRun(4)
		g.emit("VLD1.P\t64(R0), %s", list(r, 4, "H8"))
		n[i], n[i+1], n[i+2], n[i+3] = r, r+1, r+2, r+3
	}
	g.quantZig(n)
	g.emit("MOVD\tR6, ret+24(FP)")
	g.emit("RET")
	g.allFree()
	g.text("\n")
}

const header = `// The MIT License (MIT)
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

// Code generated by asmgen.go; DO NOT EDIT.

//go:build arm64 && !purego

#include "go_asm.h"
#include "textflag.h"

// NEON kernels of package dct; see asmgen.go for how they are derived from
// dct.go and dct_arm64.go for what memory each one reads and writes.

// Instructions the Go assembler lacks, as raw encodings (checked against
// clang; d, n, m are register numbers).

// MUL, MLA, MLS Vd.4S, Vn.4S, Vm.S[i]
#define MULE(d, n, m, i) WORD $(0x4F808000 | (((i)&1)<<21) | (((i)>>1)<<11) | ((m)<<16) | ((n)<<5) | (d))
#define MLAE(d, n, m, i) WORD $(0x6F800000 | (((i)&1)<<21) | (((i)>>1)<<11) | ((m)<<16) | ((n)<<5) | (d))
#define MLSE(d, n, m, i) WORD $(0x6F804000 | (((i)&1)<<21) | (((i)>>1)<<11) | ((m)<<16) | ((n)<<5) | (d))
// ADDHN, SUBHN Vd.4H, Vn.4S, Vm.4S and ADDHN2, SUBHN2 Vd.8H, Vn.4S, Vm.4S
#define ADDHN(d, n, m) WORD $(0x0E604000 | ((m)<<16) | ((n)<<5) | (d))
#define ADDHN2(d, n, m) WORD $(0x4E604000 | ((m)<<16) | ((n)<<5) | (d))
#define SUBHN(d, n, m) WORD $(0x0E606000 | ((m)<<16) | ((n)<<5) | (d))
#define SUBHN2(d, n, m) WORD $(0x4E606000 | ((m)<<16) | ((n)<<5) | (d))
// SQSHRN Vd.8B, Vn.8H, #2 and SQSHRN2 Vd.16B, Vn.8H, #2
#define SQSHRN8B(d, n) WORD $(0x0F0E9400 | ((n)<<5) | (d))
#define SQSHRN16B(d, n) WORD $(0x4F0E9400 | ((n)<<5) | (d))
// UADDL, USUBL Vd.8H, Vn.8B, Vm.8B
#define UADDL8H(d, n, m) WORD $(0x2E200000 | ((m)<<16) | ((n)<<5) | (d))
#define USUBL8H(d, n, m) WORD $(0x2E202000 | ((m)<<16) | ((n)<<5) | (d))
// SQDMULH, SABA Vd.8H, Vn.8H, Vm.8H
#define SQDMULH8H(d, n, m) WORD $(0x4E60B400 | ((m)<<16) | ((n)<<5) | (d))
#define SABA8H(d, n, m) WORD $(0x4E607C00 | ((m)<<16) | ((n)<<5) | (d))

`

func main() {
	out := flag.String("o", "dct_arm64.s", "output file")
	flag.Parse()
	var g gen
	g.text(header)
	g.text("// idctConst: box constants and the like, see genIDCT.\n")
	g.data32("idctConst", idctConst...)
	g.text("// fdctConst: box constants and the like, see genFDCT.\n")
	g.data32("fdctConst", fdctConst...)
	tabs, cms := zigTables()
	g.text("// fdctZig: TBL/TBX indices of the zig-zag reordering, then the bitmap mask.\n")
	g.data8("fdctZig", cms, tabs...)

	g.text(idctDoc)
	g.genIDCT(true)
	g.genIDCT(false)
	g.text(fdctDoc)
	g.genFDCT()
	g.genQuant()

	src := strings.TrimRight(g.b.String(), "\n") + "\n"
	if err := os.WriteFile(*out, []byte(src), 0o666); err != nil {
		log.Fatal(err)
	}
}

const idctDoc = `// func dequantIDCTStoreNEON(b, qt *Block, dst *byte, stride int)
// func dequantIDCTNEON(b, qt *Block)
//
// b[i] *= qt[i], then idctRows and idctCols of dct.go on 32-bit lanes, then
// either the clamped store to dst or the final >>18 back into b.
//
// LD4 loads the block so that each register holds rows 2i and 2i+1 of two
// columns j and j+4; UZP1/UZP2 then give the column vectors of rows 0-3 and
// rows 4-7 (lane = row) that idctRows needs. qt is loaded the same way, so
// the multiply is lane-wise. Four TRN1/TRN2 4x4 transposes turn idctRows'
// output into the row vectors (lane = column) of idctCols. (Transposing
// through memory, ST1 of each quarter and LD4 back, is fewer instructions
// but slower on Apple cores: the store-to-load latency is on the critical
// path.)
//
// The rounding term x0 += 1<<19 of idctCols (added to row 0 of the
// intermediate block) is folded into the row pass: adding 4 to coefficient 0
// of row 0 before its <<17 adds 1<<19 (mod 2^32) to x0, and from there to
// every output of row 0 through additions only, which is the same int32
// value.
//
// If the 63 AC coefficients are zero (before dequantization) every output
// is the same value, computed in general registers.

`

const fdctDoc = `// func fdctQuantNEON(src *byte, stride int, t *Quant, out *[BlockSize]int16) uint64
//
// FDCT (fdctCols, fdctRows of dct.go), image/jpeg's rounded division by
// 8*Q[zig] and the zig-zag reordering, returning the non-zero bitmap.
//
// fdctCols works on rows (lane = column): stage 1 widens the samples with
// UADDL/USUBL, the even part stays in 16-bit lanes until its box (all those
// values are below 2^11), and everything else is in 32-bit lanes. Four 4x4
// transposes (TRN1/TRN2) follow, so that fdctRows works on columns (lane =
// row). Its outputs v satisfy
// |v| < 2^31 - 2^17 (TestFDCTBound), so (v + 1<<17) >> 18 never wraps and
// equals SRSHR(v>>16, #2) in int16; the result a is in [-8192, 8191].
//
// Quantization (quantZig): with x = |a| + 4*Q (< 2^14) and the per-coefficient
// magic numbers of initVec, (x*mul) >> (15+k) = x / (8*Q) exactly
// (TestQuantExhaustive); SQDMULH computes (2*x*mul) >> 16 and USHL by -k does
// the rest. The sign is restored with (q^s)-s, s = a>>15.
//
// The quantized coefficients are column vectors (lane = row), in two runs of
// four registers; TBL/TBX with fdctZig indices gather the zig-zag order, and
// the bitmap is built like libjpeg-turbo's jchuff-neon.c: CMTST, narrow,
// AND with {0x80, 0x40, ..., 0x01}, three ADDP rounds and a byte reverse.

`
