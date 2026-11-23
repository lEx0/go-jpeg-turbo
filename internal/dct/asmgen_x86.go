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

// asmgen_x86 writes dct_amd64.s, the x86 kernels of package dct (SSE4.1,
// AVX2 and AVX-512 levels):
//
//	go run asmgen_x86.go [-o file]
//
// Like asmgen.go for NEON, the kernels are straight-line translations of
// dct.go into 32-bit vector lanes: every Go int32 product, sum, difference
// and shift becomes the same x86 operation (PMULLD, PADDD, PSUBD, PSLLD,
// PSRAD wrap and round exactly like Go), so the results are bit-exact for
// every input, including inputs whose int32 arithmetic overflows. dctBox is
// computed as dct.go writes it (three multiplications). The only rewrites
// are identities of the ring of int32 values with wrapping arithmetic
// (Z/2^32), each stated where it is used.
//
// The generator does the register allocation (lowest free register first)
// and writes each pass once for the 2-operand SSE forms and the 3-operand
// VEX forms (op3 inserts the MOVO copies SSE needs).
package main

import (
	"bytes"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
)

// reg is a vector register number.
type reg int

const none reg = -1

// mem is a memory operand in Go assembler syntax.
type mem string

type gen struct {
	b     bytes.Buffer
	used  [32]bool
	nregs int
	vex   bool // VEX three-operand forms (AVX2, AVX-512 levels)
	w     byte // register prefix of the current vector width: 'X' or 'Y'
	peak  int
}

func (g *gen) emit(format string, args ...any) {
	fmt.Fprintf(&g.b, "\t"+format+"\n", args...)
}

func (g *gen) text(s string) { g.b.WriteString(s) }

func (g *gen) label(s string) { g.b.WriteString(s + ":\n") }

// level prepares the generator for the kernels of one level.
func (g *gen) level(vex bool, w byte) {
	g.allFree()
	g.vex, g.w, g.nregs, g.peak = vex, w, 16, 0
}

func (g *gen) alloc() reg {
	for i := 0; i < g.nregs; i++ {
		if !g.used[i] {
			g.used[i] = true
			n := 0
			for _, u := range g.used {
				if u {
					n++
				}
			}
			g.peak = max(g.peak, n)
			return reg(i)
		}
	}
	panic("out of vector registers")
}

func (g *gen) free(rs ...reg) {
	for _, r := range rs {
		if !g.used[r] {
			panic(fmt.Sprintf("register %d freed twice", r))
		}
		g.used[r] = false
	}
}

func (g *gen) allFree() {
	for i, u := range g.used {
		if u {
			panic(fmt.Sprintf("register %d still allocated", i))
		}
	}
}

func (g *gen) v(r reg) string { return fmt.Sprintf("%c%d", g.w, int(r)) }

func xr(r reg) string { return fmt.Sprintf("X%d", int(r)) }
func yr(r reg) string { return fmt.Sprintf("Y%d", int(r)) }
func zr(r reg) string { return fmt.Sprintf("Z%d", int(r)) }

func (g *gen) arg(o any) string {
	switch o := o.(type) {
	case reg:
		return g.v(o)
	case mem:
		return string(o)
	}
	panic(fmt.Sprintf("bad operand %v", o))
}

// op is a vector operation: its SSE and VEX mnemonics.
type op struct {
	sse, vex string
	comm     bool
}

var (
	add32   = op{"PADDL", "VPADDD", true}
	sub32   = op{"PSUBL", "VPSUBD", false}
	mul32   = op{"PMULLD", "VPMULLD", true}
	or      = op{"POR", "VPOR", true}
	and     = op{"PAND", "VPAND", true}
	xor     = op{"PXOR", "VPXOR", true}
	unlo32  = op{"PUNPCKLLQ", "VPUNPCKLDQ", false}
	unhi32  = op{"PUNPCKHLQ", "VPUNPCKHDQ", false}
	unlo64  = op{"PUNPCKLQDQ", "VPUNPCKLQDQ", false}
	unhi64  = op{"PUNPCKHQDQ", "VPUNPCKHQDQ", false}
	packs32 = op{"PACKSSLW", "VPACKSSDW", false}
	packs16 = op{"PACKSSWB", "VPACKSSWB", false}
	shufb   = op{"PSHUFB", "VPSHUFB", false}
	add16   = op{"PADDW", "VPADDW", true}
	mulhu16 = op{"PMULHUW", "VPMULHUW", true}
	sign16  = op{"PSIGNW", "VPSIGNW", false}
	cmpeq8  = op{"PCMPEQB", "VPCMPEQB", true}

	sll32 = op{"PSLLL", "VPSLLD", false}
	sra32 = op{"PSRAL", "VPSRAD", false}
)

// op3 emits d = a OP b (b a register or a memory operand; with SSE a
// memory operand must be 16-byte aligned, which RODATA is).
func (g *gen) op3(o op, d, a reg, b any) {
	if g.vex {
		g.emit("%s\t%s, %s, %s", o.vex, g.arg(b), g.v(a), g.v(d))
		return
	}
	if d != a {
		if br, ok := b.(reg); ok && br == d {
			if !o.comm {
				panic("op3: SSE destination is the second operand of " + o.sse)
			}
			g.emit("%s\t%s, %s", o.sse, g.v(a), g.v(d))
			return
		}
		g.emit("MOVO\t%s, %s", g.v(a), g.v(d))
	}
	g.emit("%s\t%s, %s", o.sse, g.arg(b), g.v(d))
}

// sh emits d = a SHIFT k.
func (g *gen) sh(o op, d, a reg, k int) {
	if g.vex {
		g.emit("%s\t$%d, %s, %s", o.vex, k, g.v(a), g.v(d))
		return
	}
	if d != a {
		g.emit("MOVO\t%s, %s", g.v(a), g.v(d))
	}
	g.emit("%s\t$%d, %s", o.sse, k, g.v(d))
}

func (g *gen) mov(d, a reg) {
	if g.vex {
		g.emit("VMOVDQA\t%s, %s", g.v(a), g.v(d))
	} else {
		g.emit("MOVO\t%s, %s", g.v(a), g.v(d))
	}
}

func (g *gen) copyOf(a reg) reg {
	d := g.alloc()
	g.mov(d, a)
	return d
}

func (g *gen) load(d reg, m mem) {
	if g.vex {
		g.emit("VMOVDQU\t%s, %s", m, g.v(d))
	} else {
		g.emit("MOVOU\t%s, %s", m, g.v(d))
	}
}

func (g *gen) store(r reg, m mem) {
	if g.vex {
		g.emit("VMOVDQU\t%s, %s", g.v(r), m)
	} else {
		g.emit("MOVOU\t%s, %s", g.v(r), m)
	}
}

func (g *gen) abs16(d, a reg) {
	if g.vex {
		g.emit("VPABSW\t%s, %s", g.v(a), g.v(d))
	} else {
		g.emit("PABSW\t%s, %s", g.v(a), g.v(d))
	}
}

func (g *gen) ptest(a, b reg) {
	if g.vex {
		g.emit("VPTEST\t%s, %s", g.v(b), g.v(a))
	} else {
		g.emit("PTEST\t%s, %s", g.v(b), g.v(a))
	}
}

// bfly is the butterfly x+y, x-y; it frees one of x, y.
func (g *gen) bfly(x, y reg) (reg, reg) {
	s := g.alloc()
	g.op3(add32, s, x, y)
	if g.vex {
		g.op3(sub32, y, x, y)
		g.free(x)
		return s, y
	}
	g.op3(sub32, x, x, y)
	g.free(y)
	return s, x
}

// bflyKeep is bfly that keeps x.
func (g *gen) bflyKeep(x, y reg) (reg, reg) {
	s := g.alloc()
	g.op3(add32, s, x, y)
	if g.vex {
		g.op3(sub32, y, x, y)
		return s, y
	}
	d := g.alloc()
	g.op3(sub32, d, x, y)
	g.free(y)
	return s, d
}

// boxK holds the constants of dctBox(x0, x1, kcos, ksin): kcos, ksin-kcos,
// kcos+ksin.
type boxK struct{ c, d, p mem }

// box is dctBox(x0, x1, kcos, ksin) as dct.go computes it: ksum =
// kcos*(x0+x1), y0 = ksum + (ksin-kcos)*x1, y1 = ksum - (kcos+ksin)*x0.
// With round, round is added to ksum (and so to both results). (The
// four-multiplication form kcos*x0 + ksin*x1, kcos*x1 - ksin*x0, the same
// values in Z/2^32 and one step shorter, measured no faster on Zen 4.)
func (g *gen) box(x0, x1 reg, k boxK, round any) (y0, y1 reg) {
	t := g.alloc()
	g.op3(add32, t, x0, x1)
	g.op3(mul32, t, t, k.c)
	if round != nil {
		g.op3(add32, t, t, round)
	}
	g.op3(mul32, x1, x1, k.d)
	g.op3(mul32, x0, x0, k.p)
	g.op3(add32, x1, x1, t)
	if g.vex {
		g.op3(sub32, x0, t, x0)
		g.free(t)
		return x1, x0
	}
	g.op3(sub32, t, t, x0)
	g.free(x0)
	return x1, t
}

// transpose4 transposes the 4x4 int32 matrices held in r (one row per
// register, per 128-bit lane) and returns their columns.
func (g *gen) transpose4(r [4]reg) [4]reg {
	t0 := g.alloc()
	g.op3(unlo32, t0, r[0], r[1])
	g.op3(unhi32, r[0], r[0], r[1])
	g.free(r[1])
	t2 := g.alloc()
	g.op3(unlo32, t2, r[2], r[3])
	g.op3(unhi32, r[2], r[2], r[3])
	g.free(r[3])
	c0 := g.alloc()
	g.op3(unlo64, c0, t0, t2)
	g.op3(unhi64, t0, t0, t2)
	g.free(t2)
	c2 := g.alloc()
	g.op3(unlo64, c2, r[0], r[2])
	g.op3(unhi64, r[0], r[0], r[2])
	g.free(r[2])
	return [4]reg{c0, t0, c2, r[0]}
}

// transpose8 transposes the 8x8 int32 matrix held in eight ymm registers.
func (g *gen) transpose8(r [8]reg) [8]reg {
	a := g.transpose4([4]reg(r[0:4]))
	b := g.transpose4([4]reg(r[4:8]))
	var c [8]reg
	for k := 0; k < 4; k++ {
		c[k] = g.alloc()
		g.emit("VPERM2I128\t$0x20, %s, %s, %s", yr(b[k]), yr(a[k]), yr(c[k]))
		g.emit("VPERM2I128\t$0x31, %s, %s, %s", yr(b[k]), yr(a[k]), yr(b[k]))
		c[k+4] = b[k]
		g.free(a[k])
	}
	return c
}

// --- constants -------------------------------------------------------------

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

// ktab is a table of 32-byte constant vectors (a RODATA symbol, so 32-byte
// aligned: the SSE kernels use its entries as memory operands).
type ktab struct {
	name  string
	off   map[string]int
	data  [][32]byte
	notes []string
}

func newKtab(name string) *ktab { return &ktab{name: name, off: map[string]int{}} }

func (t *ktab) add(name, note string, v [32]byte) {
	if _, ok := t.off[name]; ok {
		panic("duplicate constant " + name)
	}
	t.off[name] = 32 * len(t.data)
	t.data = append(t.data, v)
	t.notes = append(t.notes, note)
}

func splat32(v int32) (b [32]byte) {
	for i := 0; i < 8; i++ {
		u := uint32(v)
		b[4*i], b[4*i+1], b[4*i+2], b[4*i+3] = byte(u), byte(u>>8), byte(u>>16), byte(u>>24)
	}
	return b
}

func lanes32(v [8]int32) (b [32]byte) {
	for i, x := range v {
		u := uint32(x)
		b[4*i], b[4*i+1], b[4*i+2], b[4*i+3] = byte(u), byte(u>>8), byte(u>>16), byte(u>>24)
	}
	return b
}

func (t *ktab) addBox(name, note string, kcos, ksin int32) {
	t.add(name+"C", note+": kcos", splat32(kcos))
	t.add(name+"D", note+": ksin-kcos", splat32(ksin-kcos))
	t.add(name+"P", note+": kcos+ksin", splat32(kcos+ksin))
}

func (t *ktab) m(name string) mem {
	o, ok := t.off[name]
	if !ok {
		panic("no constant " + name)
	}
	return mem(fmt.Sprintf("%s<>+%d(SB)", t.name, o))
}

func (t *ktab) box(name string) boxK {
	return boxK{t.m(name + "C"), t.m(name + "D"), t.m(name + "P")}
}

func (g *gen) emitTab(name string, entry int, data [][]byte, notes []string) {
	off := 0
	for k, v := range data {
		if len(v) != entry {
			panic("bad table entry size")
		}
		for i := 0; i < len(v); i += 8 {
			var x uint64
			for j := 7; j >= 0; j-- {
				x = x<<8 | uint64(v[i+j])
			}
			cm := ""
			if i == 0 && k < len(notes) && notes[k] != "" {
				cm = " // " + notes[k]
			}
			g.text(fmt.Sprintf("DATA %s<>+%d(SB)/8, $0x%016X%s\n", name, off, x, cm))
			off += 8
		}
	}
	g.text(fmt.Sprintf("GLOBL %s<>(SB), RODATA|NOPTR, $%d\n\n", name, off))
}

func (g *gen) emitKtab(t *ktab) {
	var d [][]byte
	for _, v := range t.data {
		d = append(d, v[:])
	}
	g.emitTab(t.name, 32, d, t.notes)
}

var K = newKtab("dctK")

func init() {
	K.add("round", "rounding term of idctCols, added to coefficient 0 (4<<17 = 1<<19)", lanes32([8]int32{4}))
	K.add("ac0", "all but lane 0 (the AC coefficients of row 0)", lanes32([8]int32{0, -1, -1, -1, -1, -1, -1, -1}))
	K.addBox("rbox", "idctRows box(x2, x3)", c(sqrt2inv_cos6, 18), -c(sqrt2inv_sin6, 18))
	K.add("s2i8", "c(sqrt2inv, 8)", splat32(c(sqrt2inv, 8)))
	K.addBox("box3", "box(x4, x7)", c(cos3, 12), -c(sin3, 12))
	K.addBox("box1", "box(x5, x6)", c(cos1, 12), -c(sin1, 12))
	K.addBox("cbox", "idctCols box(x2, x3)", c(sqrt2inv_cos6, 12), -c(sqrt2inv_sin6, 12))
	K.add("cboxN", "idctCols -ksin", splat32(c(sqrt2inv_sin6, 12)))
	K.add("s2i14", "c(sqrt2inv, 14)", splat32(c(sqrt2inv, 14)))
	var b80 [32]byte
	for i := range b80 {
		b80[i] = 0x80
	}
	K.add("x80", "0x80 bytes", b80)
	K.add("rows", "VPERMD indices: rows of the packed bytes", lanes32([8]int32{0, 4, 1, 5, 2, 6, 3, 7}))

	K.addBox("f3c", "fdctCols box(x4, x7)", c(cos3, 18), c(sin3, 18))
	K.addBox("f1c", "fdctCols box(x5, x6)", c(cos1, 18), c(sin1, 18))
	K.addBox("f26c", "fdctCols box(x2, x3)", c(sqrt2_cos6, 18), c(sqrt2_sin6, 18))
	K.add("s2_12", "c(sqrt2, 12)", splat32(c(sqrt2, 12)))
	K.add("k1024", "128*8", splat32(1024))
	K.addBox("f3r", "fdctRows box(x4, x7)", c(cos3, 14), c(sin3, 14))
	K.addBox("f1r", "fdctRows box(x5, x6)", c(cos1, 14), c(sin1, 14))
	K.addBox("f26r", "fdctRows box(x2, x3)", c(sqrt2_cos6, 14), c(sqrt2_sin6, 14))
	K.add("s2_14", "c(sqrt2, 14)", splat32(c(sqrt2, 14)))
	K.add("r17", "1<<17, the rounding term of fdctRows", splat32(1<<17))
	var rev [32]byte
	for i := 0; i < 16; i++ {
		rev[i] = byte(15 - i)
		rev[16+i] = byte(7 - i%8 + 8*(i/8))
	}
	K.add("rev", "PSHUFB: bytes reversed (lane 0), 8-byte halves reversed (lane 1)", rev)
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

// --- inverse DCT ---------------------------------------------------------------

// idctRowPass is idctRows on rows held as eight column vectors (lane = row;
// in[k] is coefficient k of each row). It returns the eight outputs.
func (g *gen) idctRowPass(in [8]reg) [8]reg {
	x0, x7, x2, x5, x1, x6, x3, x4 := in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7]
	g.sh(sll32, x0, x0, 17)
	g.sh(sll32, x1, x1, 17)
	x0, x1 = g.bfly(x0, x1)
	x2, x3 = g.box(x2, x3, K.box("rbox"), nil)
	x1, x2 = g.bfly(x1, x2)
	x0, x3 = g.bfly(x0, x3)

	g.sh(sll32, x4, x4, 7)
	g.sh(sll32, x7, x7, 7)
	x7, x4 = g.bfly(x7, x4)
	g.op3(mul32, x6, x6, K.m("s2i8"))
	g.op3(mul32, x5, x5, K.m("s2i8"))
	x7, x5 = g.bfly(x7, x5)
	x4, x6 = g.bfly(x4, x6)
	g.sh(sra32, x4, x4, 2)
	g.sh(sra32, x7, x7, 2)
	x4, x7 = g.box(x4, x7, K.box("box3"), nil)
	g.sh(sra32, x5, x5, 2)
	g.sh(sra32, x6, x6, 2)
	x5, x6 = g.box(x5, x6, K.box("box1"), nil)

	var o [8]reg
	o[0], o[7] = g.bfly(x0, x7)
	o[1], o[6] = g.bfly(x1, x6)
	o[2], o[5] = g.bfly(x2, x5)
	o[3], o[4] = g.bfly(x3, x4)
	return o
}

// idctColPass is idctCols, without the final >>18, on columns held as
// eight row vectors (lane = column). The rounding term 1<<19 has been added
// to row 0 by the row pass (see idctDoc).
func (g *gen) idctColPass(in [8]reg) [8]reg {
	x0, x7, x2, x5, x1, x6, x3, x4 := in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7]
	x0, x1 = g.bfly(x0, x1)
	g.sh(sra32, x0, x0, 2)
	g.sh(sra32, x1, x1, 2)
	g.sh(sra32, x2, x2, 13)
	g.sh(sra32, x3, x3, 13)
	x2, x3 = g.box(x2, x3, K.box("cbox"), nil)
	x1, x2 = g.bfly(x1, x2)
	x0, x3 = g.bfly(x0, x3)

	x7, x4 = g.bfly(x7, x4)
	g.sh(sra32, x5, x5, 13)
	g.op3(mul32, x5, x5, K.m("s2i14"))
	g.sh(sra32, x6, x6, 13)
	g.op3(mul32, x6, x6, K.m("s2i14"))
	x7, x5 = g.bfly(x7, x5)
	x4, x6 = g.bfly(x4, x6)
	g.sh(sra32, x4, x4, 14)
	g.sh(sra32, x7, x7, 14)
	x4, x7 = g.box(x4, x7, K.box("box3"), nil)
	g.sh(sra32, x5, x5, 14)
	g.sh(sra32, x6, x6, 14)
	x5, x6 = g.box(x5, x6, K.box("box1"), nil)

	var y [8]reg
	y[0], y[7] = g.bfly(x0, x7)
	y[1], y[6] = g.bfly(x1, x6)
	y[2], y[5] = g.bfly(x2, x5)
	y[3], y[4] = g.bfly(x3, x4)
	return y
}

// idctColPassTop is idctColPass for rows 4-7 zero (in holds rows 0-3):
// idctCols with x1 = x6 = x3 = x4 = 0, simplified by
//
//	x0, x1 = (x0+0)>>2, (x0-0)>>2      -> both A = x0>>2
//	dctBox(u, 0) with u = x2>>13       -> kcos*u, -ksin*u
//	x7, x4 = x7+0, x7-0; x6 = 0        -> x4 = x6 = x7 (X7) after
//	                                      x4, x6 = x4+x6, x4-x6
//
// so both boxes read p = X7>>14. All of these are int32 identities (Z/2^32).
func (g *gen) idctColPassTop(in [4]reg) [8]reg {
	a, x7, u, x5 := in[0], in[1], in[2], in[3]
	g.sh(sra32, a, a, 2)
	g.sh(sra32, u, u, 13)
	y0 := g.alloc()
	g.op3(mul32, y0, u, K.m("cboxC")) // kcos*u
	g.op3(mul32, u, u, K.m("cboxN"))  // -ksin*u
	x1, x2 := g.bflyKeep(a, y0)
	x0, x3 := g.bfly(a, u)

	g.sh(sra32, x5, x5, 13)
	g.op3(mul32, x5, x5, K.m("s2i14"))
	p := g.alloc()
	g.sh(sra32, p, x7, 14)
	q, r := g.bfly(x7, x5)
	g.sh(sra32, q, q, 14)
	g.sh(sra32, r, r, 14)
	p2 := g.copyOf(p)
	x4, x7n := g.box(p, q, K.box("box3"), nil)
	x5n, x6 := g.box(r, p2, K.box("box1"), nil)

	var y [8]reg
	y[0], y[7] = g.bfly(x0, x7n)
	y[1], y[6] = g.bfly(x1, x6)
	y[2], y[5] = g.bfly(x2, x5n)
	y[3], y[4] = g.bfly(x3, x4)
	return y
}

// GPR use of the inverse DCT kernels: AX b, BX qt, DX dst, SI stride,
// R8 3*stride, R9 dst+4*stride, CX and DI scratch.
var rowAddr = [8]string{"(DX)", "(DX)(SI*1)", "(DX)(SI*2)", "(DX)(R8*1)", "(R9)", "(R9)(SI*1)", "(R9)(SI*2)", "(R9)(R8*1)"}

func (g *gen) idctProlog(name string, store bool, frame int) {
	args := 16
	if store {
		args = 32
	}
	g.text(fmt.Sprintf("TEXT ·%s(SB), NOSPLIT, $%d-%d\n", name, frame, args))
	g.emit("MOVQ\tb+0(FP), AX")
	g.emit("MOVQ\tqt+8(FP), BX")
	if store {
		g.emit("MOVQ\tdst+16(FP), DX")
		g.emit("MOVQ\tstride+24(FP), SI")
		g.emit("LEAQ\t(SI)(SI*2), R8")
		g.emit("LEAQ\t(DX)(SI*4), R9")
	}
}

func (g *gen) ret(vzero bool) {
	if vzero {
		g.emit("VZEROUPPER")
	}
	g.emit("RET")
}

// dcOnly emits the DC-only path: every output is
// ((b[0]*qt[0])<<17 + 1<<19) >> 20 (int32, wrapping; see dcValue in
// kernel_test.go).
func (g *gen) dcOnly(store bool, width int, vzero bool) {
	g.text("\n// DC only: every output is ((b[0]*qt[0])<<17 + 1<<19) >> 20 (int32,\n")
	g.text("// wrapping; see dcValue in kernel_test.go).\n")
	g.label("dconly")
	g.emit("MOVL\t(AX), CX")
	g.emit("IMULL\t(BX), CX")
	g.emit("SHLL\t$17, CX")
	g.emit("ADDL\t$0x80000, CX")
	g.emit("SARL\t$20, CX")
	if store {
		g.emit("ADDL\t$128, CX")
		g.emit("XORL\tDI, DI")
		g.emit("CMPL\tCX, $0")
		g.emit("CMOVLLT\tDI, CX")
		g.emit("MOVL\t$255, DI")
		g.emit("CMPL\tCX, DI")
		g.emit("CMOVLGT\tDI, CX")
		g.emit("MOVQ\t$0x0101010101010101, DI")
		g.emit("IMULQ\tDI, CX")
		for _, a := range rowAddr {
			g.emit("MOVQ\tCX, %s", a)
		}
		g.ret(vzero)
		return
	}
	switch width {
	case 16:
		g.emit("MOVQ\tCX, X0")
		g.emit("PSHUFL\t$0, X0, X0")
		for i := 0; i < 16; i++ {
			g.emit("MOVOU\tX0, %d(AX)", 16*i)
		}
	case 32:
		g.emit("VMOVD\tCX, X0")
		g.emit("VPBROADCASTD\tX0, Y0")
		for i := 0; i < 8; i++ {
			g.emit("VMOVDQU\tY0, %d(AX)", 32*i)
		}
	}
	g.ret(vzero)
}

// storeRows clamps eight row vectors of int32 (8 lanes = columns, before
// the >>18) to bytes and stores them to the destination rows: >>18,
// saturating packs to int8 (clamp(v, -128, 127)) and XOR 0x80 (+128 mod
// 256): clamp(v+128, 0, 255). y is freed.
func (g *gen) storeRowsAVX2(y [8]reg) {
	g.text("\n\t// >>18, clamp((v>>18) + 128) to bytes: saturating packs to int8, XOR 0x80.\n")
	for _, r := range y {
		g.sh(sra32, r, r, 18)
	}
	var w [4]reg
	for i := 0; i < 4; i++ {
		g.op3(packs32, y[2*i], y[2*i], y[2*i+1])
		g.free(y[2*i+1])
		w[i] = y[2*i]
	}
	idx := g.alloc()
	g.load(idx, K.m("rows"))
	for p := 0; p < 2; p++ {
		a, b := w[2*p], w[2*p+1]
		g.op3(packs16, a, a, b)
		g.free(b)
		g.op3(xor, a, a, K.m("x80"))
		g.emit("VPERMD\t%s, %s, %s", yr(a), yr(idx), yr(a))
		h := g.alloc()
		g.emit("VEXTRACTI128\t$1, %s, %s", yr(a), xr(h))
		g.emit("VMOVQ\t%s, %s", xr(a), rowAddr[4*p])
		g.emit("VMOVHPS\t%s, %s", xr(a), rowAddr[4*p+1])
		g.emit("VMOVQ\t%s, %s", xr(h), rowAddr[4*p+2])
		g.emit("VMOVHPS\t%s, %s", xr(h), rowAddr[4*p+3])
		g.free(a, h)
	}
	g.free(idx)
}

// storeBlockAVX2 stores eight row vectors >>18 into b (int32, row-major).
func (g *gen) storeBlockAVX2(y [8]reg) {
	g.text("\n\t// >>18 into b.\n")
	for i, r := range y {
		g.sh(sra32, r, r, 18)
		g.store(r, mem(fmt.Sprintf("%d(AX)", 32*i)))
		g.free(r)
	}
}

// idctAVX2 emits dequantIDCTStoreAVX2 (store) or dequantIDCTAVX2.
func (g *gen) idctAVX2(store bool) {
	g.level(true, 'Y')
	name := "dequantIDCTAVX2"
	if store {
		name = "dequantIDCTStoreAVX2"
	}
	g.idctProlog(name, store, 0)
	var r [8]reg
	for i := range r {
		r[i] = g.alloc()
		g.load(r[i], mem(fmt.Sprintf("%d(AX)", 32*i)))
	}
	g.text("\n\t// DC only? (OR of the 63 AC coefficients.) Rows 4-7 zero?\n")
	bot := g.alloc()
	g.op3(or, bot, r[4], r[5])
	t := g.alloc()
	g.op3(or, t, r[6], r[7])
	g.op3(or, bot, bot, t)
	g.op3(and, t, r[0], K.m("ac0"))
	g.op3(or, t, t, r[1])
	g.op3(or, t, t, r[2])
	g.op3(or, t, t, r[3])
	g.op3(or, t, t, bot)
	g.ptest(t, t)
	g.emit("JEQ\tdconly")
	g.ptest(bot, bot)
	g.emit("JEQ\ttop")
	g.free(t, bot)

	saved := g.used
	g.text("\n\t// Dequantize, idctRows (one row per lane), idctCols (one column per lane).\n")
	for i := range r {
		g.op3(mul32, r[i], r[i], mem(fmt.Sprintf("%d(BX)", 32*i)))
	}
	g.op3(add32, r[0], r[0], K.m("round"))
	cols := g.transpose8(r)
	o := g.idctRowPass(cols)
	rows := g.transpose8(o)
	y := g.idctColPass(rows)
	if store {
		g.storeRowsAVX2(y)
	} else {
		g.storeBlockAVX2(y)
	}
	g.ret(true)
	g.allFree()

	g.used = saved
	g.text("\n// Rows 4-7 of the coefficients are zero, so rows 4-7 of idctRows' output\n")
	g.text("// are zero too (every term is a product, shift or box of zeros): the row\n")
	g.text("// pass runs on rows 0-3 only (xmm), and idctCols is simplified for them\n")
	g.text("// (idctColPassTop in asmgen_x86.go).\n")
	g.label("top")
	g.free(r[4], r[5], r[6], r[7])
	for i := 0; i < 4; i++ {
		g.op3(mul32, r[i], r[i], mem(fmt.Sprintf("%d(BX)", 32*i)))
	}
	g.op3(add32, r[0], r[0], K.m("round"))
	u := g.transpose4([4]reg(r[0:4])) // u[k] = [col k | col k+4] of rows 0-3
	var in [8]reg
	for k := 0; k < 4; k++ {
		in[k] = u[k]
		in[k+4] = g.alloc()
		g.emit("VEXTRACTI128\t$1, %s, %s", yr(u[k]), xr(in[k+4]))
	}
	g.w = 'X'
	o = g.idctRowPass(in)
	g.w = 'Y'
	var p [4]reg
	for k := 0; k < 4; k++ {
		g.emit("VINSERTI128\t$1, %s, %s, %s", xr(o[k+4]), yr(o[k]), yr(o[k]))
		g.free(o[k+4])
		p[k] = o[k]
	}
	top := g.transpose4(p)
	y = g.idctColPassTop(top)
	if store {
		g.storeRowsAVX2(y)
	} else {
		g.storeBlockAVX2(y)
	}
	g.ret(true)
	g.allFree()

	g.dcOnly(store, 32, true)
	g.text("\n")
}

// --- SSE4.1 inverse DCT -----------------------------------------------------

// slot returns the memory operand of 16-byte spill slot i of the frame.
func slot(i int) mem { return mem(fmt.Sprintf("%d(SP)", 16*i)) }

// sseRowHalf dequantizes four rows (lh[r][h]: row r, columns 4h..4h+3,
// coefficients loaded from b) by the rows of qt at qoff, runs idctRows on
// them and returns their outputs as row halves (m[r][h]).
func (g *gen) sseRowHalf(lh [4][2]reg, qoff int, round bool) [4][2]reg {
	for r := 0; r < 4; r++ {
		for h := 0; h < 2; h++ {
			q := g.alloc()
			g.load(q, mem(fmt.Sprintf("%d(BX)", qoff+32*r+16*h)))
			g.op3(mul32, lh[r][h], lh[r][h], q)
			g.free(q)
		}
	}
	if round {
		g.op3(add32, lh[0][0], lh[0][0], K.m("round"))
	}
	lo := g.transpose4([4]reg{lh[0][0], lh[1][0], lh[2][0], lh[3][0]})
	hi := g.transpose4([4]reg{lh[0][1], lh[1][1], lh[2][1], lh[3][1]})
	o := g.idctRowPass([8]reg{lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]})
	a := g.transpose4([4]reg(o[0:4]))
	b := g.transpose4([4]reg(o[4:8]))
	var m [4][2]reg
	for r := 0; r < 4; r++ {
		m[r][0], m[r][1] = a[r], b[r]
	}
	return m
}

// sseStoreRows stores the clamped rows: w[h][i] holds rows 2i and 2i+1,
// columns 4h..4h+3, as int16 (after >>18). w is freed.
func (g *gen) sseStoreRows(w [2][4]reg) {
	g.text("\n\t// clamp((v>>18) + 128) to bytes: saturating pack to int8, XOR 0x80.\n")
	for p := 0; p < 2; p++ {
		a := w[0][2*p]
		g.op3(packs16, a, a, w[0][2*p+1])
		b := w[1][2*p]
		g.op3(packs16, b, b, w[1][2*p+1])
		g.free(w[0][2*p+1], w[1][2*p+1])
		e := g.alloc()
		g.op3(unlo32, e, a, b)
		g.op3(unhi32, a, a, b)
		g.free(b)
		g.op3(xor, e, e, K.m("x80"))
		g.op3(xor, a, a, K.m("x80"))
		g.emit("MOVQ\t%s, %s", xr(e), rowAddr[4*p])
		g.emit("MOVHPS\t%s, %s", xr(e), rowAddr[4*p+1])
		g.emit("MOVQ\t%s, %s", xr(a), rowAddr[4*p+2])
		g.emit("MOVHPS\t%s, %s", xr(a), rowAddr[4*p+3])
		g.free(a, e)
	}
}

// sseColTail shifts the eight column-pass outputs y (rows 0-7 of four
// columns, half h) by 18 and either packs them (store; returns the packed
// pairs) or stores them into b.
func (g *gen) sseColTail(y [8]reg, h int, store bool) [4]reg {
	for _, r := range y {
		g.sh(sra32, r, r, 18)
	}
	var w [4]reg
	if !store {
		for i, r := range y {
			g.store(r, mem(fmt.Sprintf("%d(AX)", 32*i+16*h)))
			g.free(r)
		}
		return w
	}
	for i := 0; i < 4; i++ {
		g.op3(packs32, y[2*i], y[2*i], y[2*i+1])
		g.free(y[2*i+1])
		w[i] = y[2*i]
	}
	return w
}

// idctSSE emits dequantIDCTStoreSSE41 (store) or dequantIDCTSSE41.
func (g *gen) idctSSE(store bool) {
	g.level(false, 'X')
	name := "dequantIDCTSSE41"
	if store {
		name = "dequantIDCTStoreSSE41"
	}
	g.idctProlog(name, store, 256)

	g.text("\n\t// Rows 4-7 zero? DC only? (OR of the 63 AC coefficients.)\n")
	bot := g.alloc()
	g.load(bot, "128(AX)")
	for i := 1; i < 8; i++ {
		t := g.alloc()
		g.load(t, mem(fmt.Sprintf("%d(AX)", 128+16*i)))
		g.op3(or, bot, bot, t)
		g.free(t)
	}
	var lh [4][2]reg
	for r := 0; r < 4; r++ {
		for h := 0; h < 2; h++ {
			lh[r][h] = g.alloc()
			g.load(lh[r][h], mem(fmt.Sprintf("%d(AX)", 32*r+16*h)))
		}
	}
	all := g.alloc()
	g.op3(and, all, lh[0][0], K.m("ac0"))
	g.op3(or, all, all, lh[0][1])
	for r := 1; r < 4; r++ {
		for h := 0; h < 2; h++ {
			g.op3(or, all, all, lh[r][h])
		}
	}
	g.op3(or, all, all, bot)
	g.ptest(all, all)
	g.emit("JEQ\tdconly")
	g.ptest(bot, bot)
	g.emit("JEQ\ttop")
	g.free(all, bot)
	saved := g.used

	g.text("\n\t// Rows 0-3: dequantize, idctRows (one row per lane), into the frame\n")
	g.text("\t// as row halves (slot 2r+h: row r, columns 4h..4h+3).\n")
	m := g.sseRowHalf(lh, 0, true)
	for r := 0; r < 4; r++ {
		for h := 0; h < 2; h++ {
			g.store(m[r][h], slot(2*r+h))
			g.free(m[r][h])
		}
	}
	g.text("\n\t// Rows 4-7.\n")
	var lh2 [4][2]reg
	for r := 0; r < 4; r++ {
		for h := 0; h < 2; h++ {
			lh2[r][h] = g.alloc()
			g.load(lh2[r][h], mem(fmt.Sprintf("%d(AX)", 128+32*r+16*h)))
		}
	}
	m = g.sseRowHalf(lh2, 128, false)
	for r := 0; r < 4; r++ {
		for h := 0; h < 2; h++ {
			g.store(m[r][h], slot(8+2*r+h))
			g.free(m[r][h])
		}
	}
	var w [2][4]reg
	for h := 0; h < 2; h++ {
		g.text(fmt.Sprintf("\n\t// idctCols, columns %d-%d (one column per lane).\n", 4*h, 4*h+3))
		var in [8]reg
		for r := 0; r < 8; r++ {
			in[r] = g.alloc()
			g.load(in[r], slot(2*r+h))
		}
		w[h] = g.sseColTail(g.idctColPass(in), h, store)
	}
	if store {
		g.sseStoreRows(w)
	}
	g.ret(false)
	g.allFree()

	g.used = saved
	g.text("\n// Rows 4-7 of the coefficients are zero, so rows 4-7 of idctRows' output\n")
	g.text("// are zero too: skip their dequantization and row pass, and simplify\n")
	g.text("// idctCols for them (idctColPassTop in asmgen_x86.go).\n")
	g.label("top")
	m = g.sseRowHalf(lh, 0, true)
	for h := 0; h < 2; h++ {
		g.text(fmt.Sprintf("\n\t// idctCols, columns %d-%d.\n", 4*h, 4*h+3))
		y := g.idctColPassTop([4]reg{m[0][h], m[1][h], m[2][h], m[3][h]})
		w[h] = g.sseColTail(y, h, store)
	}
	if store {
		g.sseStoreRows(w)
	}
	g.ret(false)
	g.allFree()

	g.dcOnly(store, 16, false)
	g.text("\n")
}

// --- forward DCT ---------------------------------------------------------------

// fdctColPass is fdctCols on rows held as eight row vectors (lane = column).
// It returns the output rows 0-7.
func (g *gen) fdctColPass(x [8]reg) [8]reg {
	x0, x1, x2, x3, x4, x5, x6, x7 := x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
	x0, x7 = g.bfly(x0, x7)
	x1, x6 = g.bfly(x1, x6)
	x2, x5 = g.bfly(x2, x5)
	x3, x4 = g.bfly(x3, x4)
	x4, x7 = g.box(x4, x7, K.box("f3c"), nil)
	x5, x6 = g.box(x5, x6, K.box("f1c"), nil)
	x0, x3 = g.bfly(x0, x3)
	x1, x2 = g.bfly(x1, x2)
	x2, x3 = g.box(x2, x3, K.box("f26c"), nil)
	x0, x1 = g.bfly(x0, x1)
	g.op3(sub32, x0, x0, K.m("k1024"))
	g.sh(sll32, x0, x0, 18)
	g.sh(sll32, x1, x1, 18)
	x4, x6 = g.bfly(x4, x6)
	x7, x5 = g.bfly(x7, x5)
	g.sh(sra32, x5, x5, 12)
	g.op3(mul32, x5, x5, K.m("s2_12"))
	g.sh(sra32, x6, x6, 12)
	g.op3(mul32, x6, x6, K.m("s2_12"))
	x7, x4 = g.bfly(x7, x4)
	return [8]reg{x0, x7, x2, x5, x1, x6, x3, x4}
}

// fdctRowPass is fdctRows on rows held as eight column vectors (lane =
// row), including the final cut (v + 1<<17) >> 18. It returns the output
// columns 0-7 in natural order. The rounding term is added where it reaches
// both outputs of a final butterfly or box (x0, x7, ksum of box(x2, x3)),
// and to x5, x6 alone: an int32 identity (Z/2^32).
func (g *gen) fdctRowPass(x [8]reg) [8]reg {
	x0, x1, x2, x3, x4, x5, x6, x7 := x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
	r17 := K.m("r17")
	x0, x7 = g.bfly(x0, x7)
	x1, x6 = g.bfly(x1, x6)
	x2, x5 = g.bfly(x2, x5)
	x3, x4 = g.bfly(x3, x4)
	g.sh(sra32, x4, x4, 14)
	g.sh(sra32, x7, x7, 14)
	x4, x7 = g.box(x4, x7, K.box("f3r"), nil)
	g.sh(sra32, x5, x5, 14)
	g.sh(sra32, x6, x6, 14)
	x5, x6 = g.box(x5, x6, K.box("f1r"), nil)
	x0, x3 = g.bfly(x0, x3)
	x1, x2 = g.bfly(x1, x2)
	g.sh(sra32, x2, x2, 14)
	g.sh(sra32, x3, x3, 14)
	x2, x3 = g.box(x2, x3, K.box("f26r"), r17)
	g.op3(add32, x0, x0, r17)
	x0, x1 = g.bfly(x0, x1)
	x4, x6 = g.bfly(x4, x6)
	x7, x5 = g.bfly(x7, x5)
	g.sh(sra32, x5, x5, 14)
	g.op3(mul32, x5, x5, K.m("s2_14"))
	g.op3(add32, x5, x5, r17)
	g.sh(sra32, x6, x6, 14)
	g.op3(mul32, x6, x6, K.m("s2_14"))
	g.op3(add32, x6, x6, r17)
	g.op3(add32, x7, x7, r17)
	x7, x4 = g.bfly(x7, x4)
	o := [8]reg{x0, x7, x2, x5, x1, x6, x3, x4}
	for _, r := range o {
		g.sh(sra32, r, r, 18)
	}
	return o
}

// quantize quantizes the int16 lanes of a in place: with x = |a| + 4q,
// t = (x*mul) >> 16 (PMULHUW), y = (t*scale) >> 16 = t >> s, and the sign
// of a (PSIGNW: 0 for a = 0). The table holds bias, mul, scale (and shift)
// vectors at tab, tab+128, tab+256 (and tab+384) bytes.
func (g *gen) quantize(a reg, tab func(k int) mem) reg {
	t := g.alloc()
	g.abs16(t, a)
	if g.vex {
		g.op3(add16, t, t, tab(0))
		g.op3(mulhu16, t, t, tab(1))
		g.op3(mulhu16, t, t, tab(2))
		g.op3(sign16, a, t, a)
		g.free(t)
		return a
	}
	u := g.alloc()
	for k, o := range []op{add16, mulhu16, mulhu16} {
		g.load(u, tab(k))
		g.op3(o, t, t, u)
	}
	g.free(u)
	g.op3(sign16, t, t, a)
	g.free(a)
	return t
}

// Coefficient layouts of the quantization (see initVec in dct_amd64.go).
// Layout 0 (SSE4.1, AVX-512) word i: 8-word block b = i/8 holds SSE
// register j = sseOrder[b], whose word w is (row 4*(j/4) + w%4, column
// 2*(j%4) + w/4). Layout 1 (AVX2) word i: register a = i/16, lane l =
// (i%16)/8, w = i%8: (row 4l + w%4, column a or 7-a for w >= 4).
var sseOrder = [8]int{0, 4, 1, 5, 2, 6, 3, 7}

func sseCoef(j, w int) int { return 8*(4*(j/4)+w%4) + 2*(j%4) + w/4 }

func layoutCoef(l, i int) int {
	if l == 0 {
		return sseCoef(sseOrder[i/8], i%8)
	}
	a, ln, w := i/16, (i%16)/8, i%8
	col := a
	if w >= 4 {
		col = 7 - a
	}
	return 8*(4*ln+w%4) + col
}

// zigzag index tables, built while generating.
var zigSSE, zigAVX2 [][]byte

func zigIdx(tabs *[][]byte, v []byte) int {
	*tabs = append(*tabs, v)
	return len(*tabs) - 1
}

// sseZig emits the zig-zag reordering of the eight quantized SSE registers
// s (s[j] as described at sseCoef), the stores to (DX) and the bitmap into
// CX. s is freed.
func (g *gen) sseZig(s [8]reg) {
	g.text("\n\t// Zig-zag order: PSHUFB of each source register, OR.\n")
	zero := g.alloc()
	g.op3(xor, zero, zero, zero)
	var out [2]reg
	g.emit("XORL\tCX, CX")
	for m := 0; m < 8; m++ {
		acc := none
		for j := 0; j < 8; j++ {
			var idx [16]byte
			n := 0
			for l := 0; l < 8; l++ {
				co := unzig[8*m+l]
				idx[2*l], idx[2*l+1] = 0x80, 0x80
				for w := 0; w < 8; w++ {
					if sseCoef(j, w) == co {
						idx[2*l], idx[2*l+1] = byte(2*w), byte(2*w+1)
						n++
					}
				}
			}
			if n == 0 {
				continue
			}
			k := zigIdx(&zigSSE, idx[:])
			t := g.alloc()
			g.op3(shufb, t, s[j], mem(fmt.Sprintf("zigSSE<>+%d(SB)", 16*k)))
			if acc == none {
				acc = t
			} else {
				g.op3(or, acc, acc, t)
				g.free(t)
			}
		}
		g.store(acc, mem(fmt.Sprintf("%d(DX)", 16*m)))
		out[m%2] = acc
		if m%2 == 1 {
			// Bitmap of zig 16k..16k+15: pack (non-zero stays non-zero),
			// compare with zero, reverse the bytes, PMOVMSKB: bit j is set
			// iff zig 16k+15-j is zero, which is bit 48-16k+j of the result.
			a := out[0]
			g.op3(packs16, a, a, out[1])
			g.free(out[1])
			g.op3(cmpeq8, a, a, zero)
			g.op3(shufb, a, a, K.m("rev"))
			g.emit("PMOVMSKB\t%s, DI", xr(a))
			g.free(a)
			if sh := 48 - 16*(m/2); sh > 0 {
				g.emit("SHLQ\t$%d, DI", sh)
			}
			g.emit("ORQ\tDI, CX")
		}
	}
	g.free(s[:]...)
	g.free(zero)
	g.emit("NOTQ\tCX")
}

// sseQuantPart packs the eight column outputs o (natural columns, rows
// 4p..4p+3, int32) into the four SSE registers s[4p..4p+3] and quantizes
// them (DI points at the table of layout 0).
func (g *gen) sseQuantPart(o [8]reg, p int) [4]reg {
	var s [4]reg
	for k := 0; k < 4; k++ {
		g.op3(packs32, o[2*k], o[2*k], o[2*k+1])
		g.free(o[2*k+1])
		j := 4*p + k
		pos := 0
		for b, jj := range sseOrder {
			if jj == j {
				pos = b
			}
		}
		s[k] = g.quantize(o[2*k], func(kind int) mem {
			return mem(fmt.Sprintf("%d(DI)", 128*kind+16*pos))
		})
	}
	return s
}

// GPR use of the forward DCT kernels: AX src, SI stride, DI t's table,
// DX out, R8 3*stride, R9 src+4*stride, CX the bitmap.
var srcAddr = [8]string{"(AX)", "(AX)(SI*1)", "(AX)(SI*2)", "(AX)(R8*1)", "(R9)", "(R9)(SI*1)", "(R9)(SI*2)", "(R9)(R8*1)"}

func (g *gen) fdctProlog(name string, quant bool, frame int, layout int) {
	if quant {
		g.text(fmt.Sprintf("TEXT ·%s(SB), NOSPLIT, $%d-32\n", name, frame))
		g.emit("MOVQ\tc+0(FP), AX")
		g.emit("MOVQ\tt+8(FP), DI")
		g.emit("MOVQ\tout+16(FP), DX")
	} else {
		g.text(fmt.Sprintf("TEXT ·%s(SB), NOSPLIT, $%d-40\n", name, frame))
		g.emit("MOVQ\tsrc+0(FP), AX")
		g.emit("MOVQ\tstride+8(FP), SI")
		g.emit("MOVQ\tt+16(FP), DI")
		g.emit("MOVQ\tout+24(FP), DX")
		g.emit("LEAQ\t(SI)(SI*2), R8")
		g.emit("LEAQ\t(AX)(SI*4), R9")
	}
	g.emit("ADDQ\t$(Quant_xtab+%d), DI", 512*layout)
}

func (g *gen) fdctEpilog(quant, vzero bool) {
	if quant {
		g.emit("MOVQ\tCX, ret+24(FP)")
	} else {
		g.emit("MOVQ\tCX, ret+32(FP)")
	}
	g.ret(vzero)
	g.allFree()
	g.text("\n")
}

// fdctSSE emits fdctQuantSSE41, or with quant quantSSE41 (the quantization
// alone, from c: column c of the cut values at c[8c:8c+8], int32).
func (g *gen) fdctSSE(quant bool) {
	g.level(false, 'X')
	if quant {
		g.fdctProlog("quantSSE41", true, 0, 0)
		var s [8]reg
		for p := 0; p < 2; p++ {
			var o [8]reg
			for col := 0; col < 8; col++ {
				o[col] = g.alloc()
				g.load(o[col], mem(fmt.Sprintf("%d(AX)", 32*col+16*p)))
			}
			q := g.sseQuantPart(o, p)
			copy(s[4*p:], q[:])
		}
		g.sseZig(s)
		g.fdctEpilog(true, false)
		return
	}
	g.fdctProlog("fdctQuantSSE41", false, 256, 0)
	for h := 0; h < 2; h++ {
		g.text(fmt.Sprintf("\n\t// fdctCols, columns %d-%d (one column per lane), into the frame as\n", 4*h, 4*h+3))
		g.text("\t// column halves (slot 2c+p: column c, rows 4p..4p+3).\n")
		var x [8]reg
		for r := 0; r < 8; r++ {
			x[r] = g.alloc()
			a := srcAddr[r]
			if h == 1 {
				a = "4" + a
			}
			g.emit("PMOVZXBD\t%s, %s", a, xr(x[r]))
		}
		b := g.fdctColPass(x)
		for p := 0; p < 2; p++ {
			cs := g.transpose4([4]reg(b[4*p : 4*p+4]))
			for k, r := range cs {
				g.store(r, slot(2*(4*h+k)+p))
				g.free(r)
			}
		}
	}
	var s [8]reg
	for p := 0; p < 2; p++ {
		g.text(fmt.Sprintf("\n\t// fdctRows, rows %d-%d (one row per lane), quantization.\n", 4*p, 4*p+3))
		var x [8]reg
		for col := 0; col < 8; col++ {
			x[col] = g.alloc()
			g.load(x[col], slot(2*col+p))
		}
		q := g.sseQuantPart(g.fdctRowPass(x), p)
		copy(s[4*p:], q[:])
	}
	g.sseZig(s)
	g.fdctEpilog(false, false)
}

// avx2Quant packs the column outputs o (natural columns, int32, eight rows)
// into P_a = pack(o_a, o_7-a), a = 0..3, and quantizes them (DI points at
// the table of layout 1).
func (g *gen) avx2Quant(o [8]reg) [4]reg {
	var p [4]reg
	for a := 0; a < 4; a++ {
		g.op3(packs32, o[a], o[a], o[7-a])
		g.free(o[7-a])
		p[a] = g.quantize(o[a], func(kind int) mem {
			return mem(fmt.Sprintf("%d(DI)", 128*kind+32*a))
		})
	}
	return p
}

// avx2Zig emits the zig-zag reordering of the four quantized AVX2 registers
// p, the stores to (DX) and the bitmap into CX. p is freed.
//
// Output register Y_m (m = 0..3) holds zig 8m..8m+7 in its low lane and zig
// 56-8m..63-8m in its high lane. The two are mirror images: coefficient (r,
// c) is at zig z iff (7-r, 7-c) is at zig 63-z. P_a holds rows 0-3 of
// columns a and 7-a in its low lane and rows 4-7 in its high lane, closed
// under the mirror, so one in-lane VPSHUFB of P_a (or of P'_a, P_a with its
// lanes swapped) serves both halves of Y_m.
func (g *gen) avx2Zig(p [4]reg) {
	var src [8]reg
	for a := 0; a < 4; a++ {
		src[a] = p[a]
		src[4+a] = g.alloc()
		g.emit("VPERMQ\t$0x4E, %s, %s", yr(p[a]), yr(src[4+a]))
	}
	// coefficient at source register k, word w (0..15)
	coef := func(k, w int) int {
		a, swap := k%4, k >= 4
		ln := w / 8
		if swap {
			ln = 1 - ln
		}
		return layoutCoef(1, 16*a+8*ln+w%8)
	}
	g.text("\n\t// Zig-zag order: VPSHUFB of each source register, OR.\n")
	var y [4]reg
	for m := 0; m < 4; m++ {
		acc := none
		for k := 0; k < 8; k++ {
			var idx [32]byte
			n := 0
			for half := 0; half < 2; half++ {
				blk := m
				if half == 1 {
					blk = 7 - m
				}
				for l := 0; l < 8; l++ {
					co := unzig[8*blk+l]
					b := 16*half + 2*l
					idx[b], idx[b+1] = 0x80, 0x80
					for w := 8 * half; w < 8*half+8; w++ {
						if coef(k, w) == co {
							idx[b], idx[b+1] = byte(2*(w%8)), byte(2*(w%8)+1)
							n++
						}
					}
				}
			}
			if n == 0 {
				continue
			}
			i := zigIdx(&zigAVX2, idx[:])
			t := g.alloc()
			g.op3(shufb, t, src[k], mem(fmt.Sprintf("zigAVX2<>+%d(SB)", 32*i)))
			if acc == none {
				acc = t
			} else {
				g.op3(or, acc, acc, t)
				g.free(t)
			}
		}
		g.emit("VMOVDQU\t%s, %d(DX)", xr(acc), 16*m)
		g.emit("VEXTRACTI128\t$1, %s, %d(DX)", yr(acc), 16*(7-m))
		y[m] = acc
	}
	g.free(src[:]...)

	g.text("\n\t// Bitmap: pack to bytes (non-zero stays non-zero), compare with zero,\n")
	g.text("\t// reverse (rev), regroup the lanes so that byte j of the two halves is\n")
	g.text("\t// zig 31-j and 63-j, VPMOVMSKB, NOT.\n")
	zero := g.alloc()
	g.op3(xor, zero, zero, zero)
	a, b := y[0], y[2]
	g.op3(packs16, a, a, y[1])
	g.op3(packs16, b, b, y[3])
	g.free(y[1], y[3])
	for _, r := range []reg{a, b} {
		g.op3(cmpeq8, r, r, zero)
		g.op3(shufb, r, r, K.m("rev"))
	}
	g.free(zero)
	hi := g.alloc()
	g.emit("VPERM2I128\t$0x20, %s, %s, %s", yr(a), yr(b), yr(hi)) // [b.lo | a.lo]
	g.emit("VPERM2I128\t$0x31, %s, %s, %s", yr(b), yr(a), yr(a))  // [a.hi | b.hi]
	g.free(b)
	g.emit("VPMOVMSKB\t%s, CX", yr(hi))
	g.emit("VPMOVMSKB\t%s, DI", yr(a))
	g.free(hi, a)
	g.emit("SHLQ\t$32, CX")
	g.emit("ORQ\tDI, CX")
	g.emit("NOTQ\tCX")
}

// fdctBodyAVX2 loads the samples and runs fdctCols and fdctRows, returning
// the cut output columns (int32, natural order).
func (g *gen) fdctBodyAVX2() [8]reg {
	var x [8]reg
	for r := range x {
		x[r] = g.alloc()
		g.emit("VPMOVZXBD\t%s, %s", srcAddr[r], yr(x[r]))
	}
	g.text("\n\t// fdctCols (one column per lane).\n")
	b := g.fdctColPass(x)
	g.text("\n\t// Transpose, fdctRows (one row per lane).\n")
	cols := g.transpose8(b)
	return g.fdctRowPass(cols)
}

func (g *gen) fdctAVX2(quant bool) {
	g.level(true, 'Y')
	var o [8]reg
	if quant {
		g.fdctProlog("quantAVX2", true, 0, 1)
		for col := range o {
			o[col] = g.alloc()
			g.load(o[col], mem(fmt.Sprintf("%d(AX)", 32*col)))
		}
	} else {
		g.fdctProlog("fdctQuantAVX2", false, 0, 1)
		o = g.fdctBodyAVX2()
	}
	g.text("\n\t// Quantize: q = sign(a) * ((|a| + 4q) * mul >> 16) * scale >> 16, see initVec.\n")
	p := g.avx2Quant(o)
	g.avx2Zig(p)
	g.fdctEpilog(quant, true)
}

// zig512 index tables: forward zig-zag (2 x 32 words) and reversed (2 x 32).
func zig512Tabs() [][]byte {
	pos := make(map[int]int)
	for i := 0; i < 64; i++ {
		pos[layoutCoef(0, i)] = i
	}
	var tabs [][]byte
	for t := 0; t < 4; t++ {
		v := make([]byte, 64)
		for l := 0; l < 32; l++ {
			var zig int
			switch t {
			case 0:
				zig = l
			case 1:
				zig = 32 + l
			case 2:
				zig = 63 - l
			case 3:
				zig = 31 - l
			}
			i := pos[unzig[zig]]
			v[2*l], v[2*l+1] = byte(i), 0
		}
		tabs = append(tabs, v)
	}
	return tabs
}

// fdctAVX512 emits fdctQuantAVX512 (or quantAVX512): the AVX2 transform,
// then the quantization in two zmm registers, VPERMI2W for the zig-zag
// order and VPTESTMW for the bitmap.
func (g *gen) fdctAVX512(quant bool) {
	g.level(true, 'Y')
	var o [8]reg
	if quant {
		g.fdctProlog("quantAVX512", true, 0, 0)
		for col := range o {
			o[col] = g.alloc()
			g.load(o[col], mem(fmt.Sprintf("%d(AX)", 32*col)))
		}
	} else {
		g.fdctProlog("fdctQuantAVX512", false, 0, 0)
		o = g.fdctBodyAVX2()
	}
	g.text("\n\t// Pack to [P01 | P23], [P45 | P67] (Pab = VPACKSSDW(o_a, o_b)), quantize\n")
	g.text("\t// (the sign restored by a masked negation).\n")
	for k := 0; k < 4; k++ {
		g.op3(packs32, o[2*k], o[2*k], o[2*k+1])
		g.free(o[2*k+1])
	}
	za, zb := o[0], o[4]
	g.emit("VINSERTI64X4\t$1, %s, %s, %s", yr(o[2]), zr(za), zr(za))
	g.emit("VINSERTI64X4\t$1, %s, %s, %s", yr(o[6]), zr(zb), zr(zb))
	g.free(o[2], o[6])
	zero := g.alloc()
	g.emit("VPXORD\t%s, %s, %s", zr(zero), zr(zero), zr(zero))
	zq := [2]reg{za, zb}
	for i, a := range zq {
		t := g.alloc()
		g.emit("VPABSW\t%s, %s", zr(a), zr(t))
		g.emit("VPADDW\t%d(DI), %s, %s", 64*i, zr(t), zr(t))
		g.emit("VPMULHUW\t%d(DI), %s, %s", 128+64*i, zr(t), zr(t))
		g.emit("VPSRLVW\t%d(DI), %s, %s", 384+64*i, zr(t), zr(t))
		g.emit("VPMOVW2M\t%s, K1", zr(a))
		g.emit("VPSUBW\t%s, %s, K1, %s", zr(t), zr(zero), zr(t))
		g.free(a)
		zq[i] = t
	}
	za, zb = zq[0], zq[1]
	g.free(zero)
	g.text("\n\t// Zig-zag order and the reversed order (for the bitmap), VPERMI2W.\n")
	var z [4]reg
	for t := 0; t < 4; t++ {
		z[t] = g.alloc()
		g.emit("VMOVDQU64\tzig512<>+%d(SB), %s", 64*t, zr(z[t]))
		g.emit("VPERMI2W\t%s, %s, %s", zr(zb), zr(za), zr(z[t]))
	}
	g.free(za, zb)
	g.emit("VMOVDQU64\t%s, (DX)", zr(z[0]))
	g.emit("VMOVDQU64\t%s, 64(DX)", zr(z[1]))
	g.text("\t// Bit j of K1 is set iff zig 63-j is non-zero, bit j of K2 iff zig 31-j.\n")
	g.emit("VPTESTMW\t%s, %s, K1", zr(z[2]), zr(z[2]))
	g.emit("VPTESTMW\t%s, %s, K2", zr(z[3]), zr(z[3]))
	g.emit("KUNPCKDQ\tK1, K2, K3")
	g.emit("KMOVQ\tK3, CX")
	g.free(z[:]...)
	g.fdctEpilog(quant, true)
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

// Code generated by asmgen_x86.go; DO NOT EDIT.

//go:build amd64 && !purego

#include "go_asm.h"
#include "textflag.h"

// x86 kernels of package dct; see asmgen_x86.go for how they are derived
// from dct.go and dct_amd64.go for what memory each one reads and writes.
// R14, R15 and BP are not used; the AVX2 and AVX-512 kernels end with
// VZEROUPPER on every path. The SSE4.1 kernels keep spills at 0(SP).

`

const idctDoc = `// func dequantIDCTStoreSSE41(b, qt *Block, dst *byte, stride int)
// func dequantIDCTSSE41(b, qt *Block)
// func dequantIDCTStoreAVX2(b, qt *Block, dst *byte, stride int)
// func dequantIDCTAVX2(b, qt *Block)
//
// b[i] *= qt[i], then idctRows and idctCols of dct.go on 32-bit lanes, then
// either the clamped store to dst or the final >>18 back into b.
//
// The row pass works on column vectors (lane = row): AVX2 transposes the
// dequantized block (8x8, VPUNPCK* and VPERM2I128), SSE4.1 does four rows
// at a time (4x4 transposes). Its output is transposed back to row vectors
// (lane = column) for the column pass, whose results are the output rows.
//
// The rounding term x0 += 1<<19 of idctCols (added to row 0 of the
// intermediate block) is folded into the row pass: adding 4 to coefficient 0
// of row 0 after the dequantization adds 1<<19 (mod 2^32) to x0 after its
// <<17, and from there to every output of row 0 through additions only,
// which is the same int32 value.
//
// If the 63 AC coefficients are zero (before dequantization) every output
// is the same value, computed in general registers. If rows 4-7 are zero,
// the row pass runs on rows 0-3 only and the column pass is simplified.

`

const fdctDoc = `// func fdctQuantSSE41(src *byte, stride int, t *Quant, out *[BlockSize]int16) uint64
// func fdctQuantAVX2(src *byte, stride int, t *Quant, out *[BlockSize]int16) uint64
// func fdctQuantAVX512(src *byte, stride int, t *Quant, out *[BlockSize]int16) uint64
// func quantSSE41(c *Block, t *Quant, out *[BlockSize]int16) uint64 (and AVX2, AVX512)
//
// FDCT (fdctCols, fdctRows of dct.go, in 32-bit lanes), image/jpeg's
// rounded division by 8*Q[zig] and the zig-zag reordering, returning the
// non-zero bitmap. fdctCols works on row vectors (lane = column), then the
// block is transposed and fdctRows works on column vectors (lane = row).
// Every int32 value of the cut a = (v + 1<<17) >> 18 is in [-8192, 8191],
// so the signed saturating packs to int16 are exact.
//
// Quantization: div(a, 8q) = sign(a) * floor((|a| + 4q) / 8q), and with
// x = |a| + 4q < 2^14 and the magic numbers of initVec (mul < 2^16, scale =
// 2^(16-s)), ((x*mul) >> 16) * scale >> 16 = (x*mul) >> (16+s) = x / 8q
// (TestQuantExhaustive). PSIGNW restores the sign (and gives 0 for a = 0).
//
// quantSSE41, quantAVX2 and quantAVX512 run the quantization alone (with the
// same packing and zig-zag code) on cut values c[8*col+row], for the tests.

`

func main() {
	out := flag.String("o", "dct_amd64.s", "output file")
	flag.Parse()
	var g gen
	g.nregs = 16
	g.text(header)

	var body gen
	body.nregs = 16
	body.text(idctDoc)
	for _, store := range []bool{true, false} {
		body.idctSSE(store)
		body.idctAVX2(store)
	}
	body.text(fdctDoc)
	body.fdctSSE(false)
	body.fdctSSE(true)
	body.fdctAVX2(false)
	body.fdctAVX2(true)
	body.fdctAVX512(false)
	body.fdctAVX512(true)

	g.text("// dctK: 32-byte constant vectors (box constants and the like).\n")
	g.emitKtab(K)
	g.text("// zigSSE: PSHUFB indices of the zig-zag reordering (SSE4.1).\n")
	g.emitTab("zigSSE", 16, zigSSE, nil)
	g.text("// zigAVX2: VPSHUFB indices of the zig-zag reordering (AVX2).\n")
	g.emitTab("zigAVX2", 32, zigAVX2, nil)
	g.text("// zig512: VPERMI2W indices: zig 0-31, 32-63, then 63..32 and 31..0.\n")
	g.emitTab("zig512", 64, zig512Tabs(), nil)
	g.b.Write(body.b.Bytes())

	src := strings.TrimRight(g.b.String(), "\n") + "\n"
	if err := os.WriteFile(*out, []byte(src), 0o666); err != nil {
		log.Fatal(err)
	}
}
