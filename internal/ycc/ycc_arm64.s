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

//go:build arm64 && !purego

#include "textflag.h"

// NEON color conversion and chroma downsampling, bit-exact with the *Ref
// functions in ycc.go (color.RGBToYCbCr, color.NRGBA.RGBA, image/jpeg's
// scale). All arithmetic is in 16-bit lanes; ycc_test.go checks every
// identity below exhaustively.
//
// color.RGBToYCbCr in 16 bits. Each 16-bit coefficient k = 256*hi + lo is
// split into bytes so that the products are 8x8->16-bit multiplies
// (UMULL/UMLAL .8B -> .8H). For a sum S = 256*H + L + bias:
//   (256*H + L + bias) >> 16 = (H + bias/256 + (L >> 8)) >> 8
// (bias is a multiple of 256). The H and L sums fit 16 bits because the
// hi and lo coefficients of each output add up to at most 255 and 256.
//
//   Y  = (19595 r + 38470 g + 7471 b + 1<<15) >> 16
//      H = 76 r + 150 g + 29 b, L = 139 r + 70 g + 47 b
//      Y = (H + (L >> 8) + 128) >> 8            = RSHRN #8 of H + (L >> 8)
//
// The chroma sums use complements r' = 255-r (NOT), which turn
// color.RGBToYCbCr's signed sums into unsigned ones:
//   -11056 r - 21712 g + 32768 b + 257<<15 = 32768 b + 11056 r' + 21712 g' + 1<<16
//    32768 r - 27440 g - 5328 b + 257<<15  = 32768 r + 27440 g' + 5328 b' + 1<<16
// so, with x = H + (L >> 8) <= 65280 and color.RGBToYCbCr's clamp (the
// sum lies in [1<<16, 1<<24], only the top can clamp):
//   Cb: H = 128 b + 43 r' + 84 g',  L = 48 r' + 208 g'
//   Cr: H = 128 r + 107 g' + 20 b', L = 48 g' + 208 b'
//   C  = min((x >> 8) + 1, 255)                 = UQADD 1 of SHRN #8 of x
//
// color.NRGBA.RGBA() >> 8 of a channel c with alpha a, ((c*257*a)/255)>>8,
// is, with x = c*a (UMULL .8B -> .8H):
//   y = x + (x >> 8)  (= floor(257 x / 256) <= 65279)
//   z = y + (y >> 8)  (<= 65533)
//   v = (z + 1) >> 8  (= floor(y / 255))        = ADDHN of z and 1
//
// The 2x2 average (a + b + c + d + 2) >> 2 is UADDLP (pairwise sums of a
// row), UADALP (plus the pairwise sums of the next row) and RSHRN #2,
// whose rounding constant is exactly the +2.
//
// Blocks are 16 pixels. Each function requires n >= 16 and processes the
// blocks at 0, 16, 32, ... and a last block at n-16, which may overlap the
// previous one: it recomputes the same values, so the outputs must not
// alias the inputs. Nothing outside [0, n) is read or written.
//
// Registers: V0-V3 the deinterleaved R, G, B, A bytes; V4-V6 R', G', B';
// V7-V10 accumulators (V4-V9 are also PREMUL16's temporaries); V11-V13
// Y, Cb, Cr; V14-V15 chroma sums (4:2:0); V16 and V31 unused; V17-V30
// constants (CONSTS, PREMUL_CONSTS). Only R0-R10 and R12 are used (no
// callee-saved registers in Go's arm64 ABI; R18, R26-R29 untouched).
//
// Instructions the Go assembler does not know are emitted with WORD; each
// macro names the instruction it encodes. The encodings were checked
// against clang's assembler.

// RSHRN Vd.8B, Vn.8H, #sh
#define RSHRN_8B(d, n, sh) WORD $(0x0F008C00 | ((16-(sh))<<16) | ((n)<<5) | (d))
// RSHRN2 Vd.16B, Vn.8H, #sh
#define RSHRN2_16B(d, n, sh) WORD $(0x4F008C00 | ((16-(sh))<<16) | ((n)<<5) | (d))
// ADDHN Vd.8B, Vn.8H, Vm.8H
#define ADDHN_8B(d, n, m) WORD $(0x0E204000 | ((m)<<16) | ((n)<<5) | (d))
// ADDHN2 Vd.16B, Vn.8H, Vm.8H
#define ADDHN2_16B(d, n, m) WORD $(0x4E204000 | ((m)<<16) | ((n)<<5) | (d))
// UADDLP Vd.8H, Vn.16B
#define UADDLP_8H(d, n) WORD $(0x6E202800 | ((n)<<5) | (d))
// UADALP Vd.8H, Vn.16B
#define UADALP_8H(d, n) WORD $(0x6E206800 | ((n)<<5) | (d))

// The byte coefficients (all lanes): V17-V22 luma hi 76 150 29 and lo
// 139 70 47; V23 43, V24 84 (Cb hi), V25 48, V26 208 (Cb and Cr lo), V27
// 107, V28 20 (Cr hi); V29 the byte 1.
#define CONSTS \
	VMOVI $76, V17.B16 \
	VMOVI $150, V18.B16 \
	VMOVI $29, V19.B16 \
	VMOVI $139, V20.B16 \
	VMOVI $70, V21.B16 \
	VMOVI $47, V22.B16 \
	VMOVI $43, V23.B16 \
	VMOVI $84, V24.B16 \
	VMOVI $48, V25.B16 \
	VMOVI $208, V26.B16 \
	VMOVI $107, V27.B16 \
	VMOVI $20, V28.B16 \
	VMOVI $1, V29.B16

// V30 = the halfword 1 (for ADDHN). Clobbers R12.
#define PREMUL_CONSTS \
	MOVD $1, R12 \
	VDUP R12, V30.H8

// YCC16 converts the 16 pixels whose R, G, B bytes are in V0, V1, V2 to
// Y in V11, Cb in V12 and Cr in V13. For each output, H is accumulated in
// V7 (pixels 0-7) and V8 (pixels 8-15) and L in V9 and V10. Clobbers
// V4-V10.
#define YCC16 \
	VNOT V0.B16, V4.B16 \
	VNOT V1.B16, V5.B16 \
	VNOT V2.B16, V6.B16 \
	VUMULL V17.B8, V0.B8, V7.H8 \
	VUMULL2 V17.B16, V0.B16, V8.H8 \
	VUMULL V20.B8, V0.B8, V9.H8 \
	VUMULL2 V20.B16, V0.B16, V10.H8 \
	VUMLAL V18.B8, V1.B8, V7.H8 \
	VUMLAL2 V18.B16, V1.B16, V8.H8 \
	VUMLAL V21.B8, V1.B8, V9.H8 \
	VUMLAL2 V21.B16, V1.B16, V10.H8 \
	VUMLAL V19.B8, V2.B8, V7.H8 \
	VUMLAL2 V19.B16, V2.B16, V8.H8 \
	VUMLAL V22.B8, V2.B8, V9.H8 \
	VUMLAL2 V22.B16, V2.B16, V10.H8 \
	VUSRA $8, V9.H8, V7.H8 \
	VUSRA $8, V10.H8, V8.H8 \
	RSHRN_8B(11, 7, 8) \
	RSHRN2_16B(11, 8, 8) \
	VUSHLL $7, V2.B8, V7.H8 \
	VUSHLL2 $7, V2.B16, V8.H8 \
	VUMULL V25.B8, V4.B8, V9.H8 \
	VUMULL2 V25.B16, V4.B16, V10.H8 \
	VUMLAL V23.B8, V4.B8, V7.H8 \
	VUMLAL2 V23.B16, V4.B16, V8.H8 \
	VUMLAL V26.B8, V5.B8, V9.H8 \
	VUMLAL2 V26.B16, V5.B16, V10.H8 \
	VUMLAL V24.B8, V5.B8, V7.H8 \
	VUMLAL2 V24.B16, V5.B16, V8.H8 \
	VUSRA $8, V9.H8, V7.H8 \
	VUSRA $8, V10.H8, V8.H8 \
	VSHRN $8, V7.H8, V12.B8 \
	VSHRN2 $8, V8.H8, V12.B16 \
	VUQADD V29.B16, V12.B16, V12.B16 \
	VUSHLL $7, V0.B8, V7.H8 \
	VUSHLL2 $7, V0.B16, V8.H8 \
	VUMULL V25.B8, V5.B8, V9.H8 \
	VUMULL2 V25.B16, V5.B16, V10.H8 \
	VUMLAL V27.B8, V5.B8, V7.H8 \
	VUMLAL2 V27.B16, V5.B16, V8.H8 \
	VUMLAL V26.B8, V6.B8, V9.H8 \
	VUMLAL2 V26.B16, V6.B16, V10.H8 \
	VUMLAL V28.B8, V6.B8, V7.H8 \
	VUMLAL2 V28.B16, V6.B16, V8.H8 \
	VUSRA $8, V9.H8, V7.H8 \
	VUSRA $8, V10.H8, V8.H8 \
	VSHRN $8, V7.H8, V13.B8 \
	VSHRN2 $8, V8.H8, V13.B16 \
	VUQADD V29.B16, V13.B16, V13.B16

// PREMUL16 replaces the R, G, B bytes in V0, V1, V2 by their 8-bit
// premultiplied values (alpha in V3). Clobbers V4-V9.
#define PREMUL16 \
	VUMULL V3.B8, V0.B8, V4.H8 \
	VUMULL2 V3.B16, V0.B16, V5.H8 \
	VUMULL V3.B8, V1.B8, V6.H8 \
	VUMULL2 V3.B16, V1.B16, V7.H8 \
	VUMULL V3.B8, V2.B8, V8.H8 \
	VUMULL2 V3.B16, V2.B16, V9.H8 \
	VUSRA $8, V4.H8, V4.H8 \
	VUSRA $8, V5.H8, V5.H8 \
	VUSRA $8, V6.H8, V6.H8 \
	VUSRA $8, V7.H8, V7.H8 \
	VUSRA $8, V8.H8, V8.H8 \
	VUSRA $8, V9.H8, V9.H8 \
	VUSRA $8, V4.H8, V4.H8 \
	VUSRA $8, V5.H8, V5.H8 \
	VUSRA $8, V6.H8, V6.H8 \
	VUSRA $8, V7.H8, V7.H8 \
	VUSRA $8, V8.H8, V8.H8 \
	VUSRA $8, V9.H8, V9.H8 \
	ADDHN_8B(0, 4, 30) \
	ADDHN2_16B(0, 5, 30) \
	ADDHN_8B(1, 6, 30) \
	ADDHN2_16B(1, 7, 30) \
	ADDHN_8B(2, 8, 30) \
	ADDHN2_16B(2, 9, 30)

// func rgbaToYCbCrNEON(rgba, y, cb, cr *byte, n int)
//
// Converts n >= 16 RGBA pixels: reads rgba[:4n], writes y[:n], cb[:n] and
// cr[:n].
TEXT ·rgbaToYCbCrNEON(SB), NOSPLIT, $0-40
	MOVD rgba+0(FP), R0
	MOVD y+8(FP), R1
	MOVD cb+16(FP), R2
	MOVD cr+24(FP), R3
	MOVD n+32(FP), R4
	CONSTS
	SUB  $16, R4, R5 // last block
	MOVD $0, R6      // block start i

rgbaloop:
	CMP  R5, R6
	CSEL GT, R5, R6, R6     // i = min(i, n-16)
	ADD  R6<<2, R0, R7
	VLD4 (R7), [V0.B16, V1.B16, V2.B16, V3.B16]
	YCC16
	ADD  R6, R1, R7
	VST1 [V11.B16], (R7)
	ADD  R6, R2, R7
	VST1 [V12.B16], (R7)
	ADD  R6, R3, R7
	VST1 [V13.B16], (R7)
	CMP  R5, R6
	BEQ  rgbadone
	ADD  $16, R6
	B    rgbaloop

rgbadone:
	RET

// func nrgbaToYCbCrNEON(nrgba, y, cb, cr *byte, n int)
//
// Converts n >= 16 NRGBA pixels: reads nrgba[:4n], writes y[:n], cb[:n]
// and cr[:n].
TEXT ·nrgbaToYCbCrNEON(SB), NOSPLIT, $0-40
	MOVD nrgba+0(FP), R0
	MOVD y+8(FP), R1
	MOVD cb+16(FP), R2
	MOVD cr+24(FP), R3
	MOVD n+32(FP), R4
	CONSTS
	PREMUL_CONSTS
	SUB  $16, R4, R5
	MOVD $0, R6

nrgbaloop:
	CMP  R5, R6
	CSEL GT, R5, R6, R6
	ADD  R6<<2, R0, R7
	VLD4 (R7), [V0.B16, V1.B16, V2.B16, V3.B16]
	PREMUL16
	YCC16
	ADD  R6, R1, R7
	VST1 [V11.B16], (R7)
	ADD  R6, R2, R7
	VST1 [V12.B16], (R7)
	ADD  R6, R3, R7
	VST1 [V13.B16], (R7)
	CMP  R5, R6
	BEQ  nrgbadone
	ADD  $16, R6
	B    nrgbaloop

nrgbadone:
	RET

// func downsample2x2NEON(r0, r1, out *byte, n int)
//
// Averages n >= 16 2x2 squares: reads r0[:2n] and r1[:2n], writes
// out[:n].
TEXT ·downsample2x2NEON(SB), NOSPLIT, $0-32
	MOVD r0+0(FP), R0
	MOVD r1+8(FP), R1
	MOVD out+16(FP), R2
	MOVD n+24(FP), R3
	SUB  $16, R3, R5
	MOVD $0, R6

dsloop:
	CMP  R5, R6
	CSEL GT, R5, R6, R6
	ADD  R6<<1, R0, R7
	VLD1 (R7), [V0.B16, V1.B16]
	ADD  R6<<1, R1, R7
	VLD1 (R7), [V2.B16, V3.B16]
	UADDLP_8H(4, 0)
	UADDLP_8H(5, 1)
	UADALP_8H(4, 2)
	UADALP_8H(5, 3)
	RSHRN_8B(6, 4, 2)
	RSHRN2_16B(6, 5, 2)
	ADD  R6, R2, R7
	VST1 [V6.B16], (R7)
	CMP  R5, R6
	BEQ  dsdone
	ADD  $16, R6
	B    dsloop

dsdone:
	RET

// The 4:2:0 loops convert the 16-pixel blocks at pixel i (R8) of both
// rows: R0/R1 the input rows, R2/R3 the luma rows, R4/R5 the chroma rows.
// LOAD420 loads the block of row ROW, STORE420Y stores its luma to row YR
// and STORE420C stores the 8 averaged chroma samples. Clobber R9, R10.
#define LOAD420(ROW) \
	ADD  R8<<2, ROW, R9 \
	VLD4 (R9), [V0.B16, V1.B16, V2.B16, V3.B16]

#define STORE420Y(YR) \
	ADD  R8, YR, R9 \
	VST1 [V11.B16], (R9)

#define STORE420C \
	RSHRN_8B(12, 14, 2) \
	RSHRN_8B(13, 15, 2) \
	LSR  $1, R8, R10 \
	ADD  R10, R4, R9 \
	VST1 [V12.B8], (R9) \
	ADD  R10, R5, R9 \
	VST1 [V13.B8], (R9)

// func rgbaToYCbCr420NEON(rgba0, rgba1, y0, y1, cb, cr *byte, n int)
//
// Converts two rows of n >= 16 RGBA pixels, n even: reads rgba0[:4n] and
// rgba1[:4n], writes y0[:n], y1[:n], cb[:n/2] and cr[:n/2].
TEXT ·rgbaToYCbCr420NEON(SB), NOSPLIT, $0-56
	MOVD rgba0+0(FP), R0
	MOVD rgba1+8(FP), R1
	MOVD y0+16(FP), R2
	MOVD y1+24(FP), R3
	MOVD cb+32(FP), R4
	MOVD cr+40(FP), R5
	MOVD n+48(FP), R6
	CONSTS
	SUB  $16, R6, R7
	MOVD $0, R8

r420loop:
	CMP  R7, R8
	CSEL GT, R7, R8, R8
	LOAD420(R0)
	YCC16
	STORE420Y(R2)
	UADDLP_8H(14, 12)
	UADDLP_8H(15, 13)
	LOAD420(R1)
	YCC16
	STORE420Y(R3)
	UADALP_8H(14, 12)
	UADALP_8H(15, 13)
	STORE420C
	CMP  R7, R8
	BEQ  r420done
	ADD  $16, R8
	B    r420loop

r420done:
	RET

// func nrgbaToYCbCr420NEON(nrgba0, nrgba1, y0, y1, cb, cr *byte, n int)
//
// Converts two rows of n >= 16 NRGBA pixels, n even: reads nrgba0[:4n]
// and nrgba1[:4n], writes y0[:n], y1[:n], cb[:n/2] and cr[:n/2].
TEXT ·nrgbaToYCbCr420NEON(SB), NOSPLIT, $0-56
	MOVD nrgba0+0(FP), R0
	MOVD nrgba1+8(FP), R1
	MOVD y0+16(FP), R2
	MOVD y1+24(FP), R3
	MOVD cb+32(FP), R4
	MOVD cr+40(FP), R5
	MOVD n+48(FP), R6
	CONSTS
	PREMUL_CONSTS
	SUB  $16, R6, R7
	MOVD $0, R8

n420loop:
	CMP  R7, R8
	CSEL GT, R7, R8, R8
	LOAD420(R0)
	PREMUL16
	YCC16
	STORE420Y(R2)
	UADDLP_8H(14, 12)
	UADDLP_8H(15, 13)
	LOAD420(R1)
	PREMUL16
	YCC16
	STORE420Y(R3)
	UADALP_8H(14, 12)
	UADALP_8H(15, 13)
	STORE420C
	CMP  R7, R8
	BEQ  n420done
	ADD  $16, R8
	B    n420loop

n420done:
	RET
