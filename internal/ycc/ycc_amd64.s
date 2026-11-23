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

#include "textflag.h"

// x86 color conversion and chroma downsampling, bit-exact with the *Ref
// functions in ycc.go (color.RGBToYCbCr, color.NRGBA.RGBA, image/jpeg's
// scale); ycc_test.go checks every identity below exhaustively.
//
// color.RGBToYCbCr in 32-bit lanes (PMADDWD, PMADDWL in Go's assembler).
// Each pixel's 32-bit lane holds two 16-bit words: P1 = (r, g) and
// P2 = (b, 256), shuffled from the RGBA bytes (P2's 256 via OR). PMADDWD
// by a pair of int16 coefficients gives k0*w0 + k1*w1 exactly. With every coefficient in int16:
//   Y  = (P1 + P1.(19594, -27066) + P2.(7471, 128)) >> 16
// where P1 read as an int32 is r + 65536 g, so the sum is
// 19595 r + 38470 g + 7471 b + 32768 (38470 does not fit int16). The
// chroma sums of color.RGBToYCbCr have even coefficients and are halved:
//   Cb = (P1.(-5528, -10856) + P2.(16384, 16448)) >> 15
//   Cr = (P1.(16384, -13720) + P2.(-2664, 16448)) >> 15
// (16448*256 = 257<<14). The halved sums lie in [1<<15, 1<<23], so Cb
// and Cr are in [1, 256]; PACKSSDW (PACKSSLW, no saturation) and
// PACKUSWB (which clamps 256 to 255, color.RGBToYCbCr's clamp) narrow
// them to bytes.
//
// color.NRGBA.RGBA() >> 8 of a channel c with alpha a, ((c*257*a)/255)>>8,
// is (c*a*33025) >> 23 = PMULHUW(PMULLW(c, a), 33025) >> 7 on the 16-bit
// words of P1 and P2 (before P2's 256 is set). For a = 255 it is c, so
// the NRGBA kernels convert a block whose alpha bytes are all 255 (both
// rows' blocks for the 4:2:0 kernels) as RGBA, without premultiplying.
//
// The 2x2 average (a + b + c + d + 2) >> 2 is PMADDUBSW by bytes 1 (the
// sums of the horizontal pairs of a row, at most 510), PADDW (plus the
// next row's) and PMULHRSW by 8192, which computes
// ((x*8192 >> 14) + 1) >> 1 = ((x >> 1) + 1) >> 1 = (x + 2) >> 2.
//
// Blocks are 16 pixels (SSE41), 32 (AVX2) or 64 (AVX512). Each function
// requires n >= its block and processes the blocks at 0, B, 2B, ... and a
// last block at n-B, which may overlap the previous one: it recomputes the
// same values, so the outputs must not alias the inputs. Nothing outside
// [0, n) is read or written.
//
// The SSE41 kernels use legacy SSE encodings only (their memory operands
// are 16-byte aligned: the linker aligns these 64-byte symbols to 32);
// the AVX2 ones use VEX encodings only; the AVX2 and AVX512 ones end with
// VZEROUPPER. Only AX, BX, CX, DX, SI, DI, R8-R12 and (AVX512) K1 are
// used.

// C64 defines a 64-byte constant repeating the quadword v; C64X2 one
// repeating the 16 bytes lo, hi.
#define C64(sym, v) \
	DATA sym+0(SB)/8, v \
	DATA sym+8(SB)/8, v \
	DATA sym+16(SB)/8, v \
	DATA sym+24(SB)/8, v \
	DATA sym+32(SB)/8, v \
	DATA sym+40(SB)/8, v \
	DATA sym+48(SB)/8, v \
	DATA sym+56(SB)/8, v \
	GLOBL sym(SB), RODATA|NOPTR, $64

#define C64X2(sym, lo, hi) \
	DATA sym+0(SB)/8, lo \
	DATA sym+8(SB)/8, hi \
	DATA sym+16(SB)/8, lo \
	DATA sym+24(SB)/8, hi \
	DATA sym+32(SB)/8, lo \
	DATA sym+40(SB)/8, hi \
	DATA sym+48(SB)/8, lo \
	DATA sym+56(SB)/8, hi \
	GLOBL sym(SB), RODATA|NOPTR, $64

// PSHUFB masks: P1 = (r, g) and P2 = (b, 0) words of each pixel, and the
// alpha in both words.
C64X2(yccShufRG<>, $0x8005800480018000, $0x800D800C80098008)
C64X2(yccShufB<>, $0x8080800680808002, $0x8080800E8080800A)
C64X2(yccShufA<>, $0x8007800780038003, $0x800F800F800B800B)
// 256 in P2's high word.
C64(yccK256<>, $0x0100000001000000)
// The alpha bytes of RGBA pixels.
C64(yccAlphaFF<>, $0xFF000000FF000000)
// PMADDWD coefficient pairs (low word first): Y (19594, -27066) and
// (7471, 128); Cb (-5528, -10856) and (16384, 16448); Cr (16384, -13720)
// and (-2664, 16448).
C64(yccY1<>, $0x96464C8A96464C8A)
C64(yccY2<>, $0x00801D2F00801D2F)
C64(yccCb1<>, $0xD598EA68D598EA68)
C64(yccCb2<>, $0x4040400040404000)
C64(yccCr1<>, $0xCA684000CA684000)
C64(yccCr2<>, $0x4040F5984040F598)
// Premultiplication factor 33025 (words), downsampling bytes 1 and
// PMULHRSW factor 8192 (words).
C64(yccK33025<>, $0x8101810181018101)
C64(yccOnes<>, $0x0101010101010101)
C64(yccK8192<>, $0x2000200020002000)

// VPERMD indices that undo the lane interleaving of the AVX2 packs: the
// packed bytes of 32 pixels hold the 4-pixel groups 0 2 4 6 1 3 5 7.
DATA yccPermD<>+0(SB)/8, $0x0000000400000000
DATA yccPermD<>+8(SB)/8, $0x0000000500000001
DATA yccPermD<>+16(SB)/8, $0x0000000600000002
DATA yccPermD<>+24(SB)/8, $0x0000000700000003
GLOBL yccPermD<>(SB), RODATA|NOPTR, $32

// PSHUFB mask putting the chroma words of the 4-pixel groups 0 2 4 6 1 3
// 5 7 (AVX2 4:2:0 kernels) in order.
C64X2(yccFix420<>, $0x0B0A030209080100, $0x0F0E07060D0C0504)

// ---- SSE4.1 ---------------------------------------------------------------

// LOAD4_SSE loads the 4 RGBA pixels at SRC as P1 (X0) and P2 (X1). X15
// holds yccShufRG and X14 yccShufB; K256 is yccK256 (register or memory).
#define LOAD4_SSE(SRC, K256) \
	MOVOU SRC, X0 \
	MOVO X0, X1 \
	PSHUFB X15, X0 \
	PSHUFB X14, X1 \
	POR K256, X1

// LOAD4N_SSE is LOAD4_SSE for NRGBA pixels: it premultiplies P1 and P2
// (KA is yccShufA). Clobbers X2.
#define LOAD4N_SSE(SRC, K256, KA) \
	MOVOU SRC, X0 \
	MOVO X0, X1 \
	MOVO X0, X2 \
	PSHUFB X15, X0 \
	PSHUFB X14, X1 \
	PSHUFB KA, X2 \
	PMULLW X2, X0 \
	PMULLW X2, X1 \
	PMULHUW yccK33025<>(SB), X0 \
	PMULHUW yccK33025<>(SB), X1 \
	PSRLW $7, X0 \
	PSRLW $7, X1 \
	POR K256, X1

// OPAQUE16_SSE sets CF if the 16 NRGBA pixels at (BASE) all have alpha
// 255. Clobbers X0, X1.
#define OPAQUE16_SSE(BASE) \
	MOVOU (BASE), X0 \
	MOVOU 16(BASE), X1 \
	PAND X1, X0 \
	MOVOU 32(BASE), X1 \
	PAND X1, X0 \
	MOVOU 48(BASE), X1 \
	PAND X1, X0 \
	PTEST yccAlphaFF<>(SB), X0

// OPAQUE16x2_SSE is OPAQUE16_SSE for the blocks at (B0) and (B1).
// Clobbers X0-X2.
#define OPAQUE16x2_SSE(B0, B1) \
	MOVOU (B0), X0 \
	MOVOU 16(B0), X1 \
	PAND X1, X0 \
	MOVOU 32(B0), X1 \
	PAND X1, X0 \
	MOVOU 48(B0), X1 \
	PAND X1, X0 \
	MOVOU (B1), X1 \
	MOVOU 16(B1), X2 \
	PAND X2, X1 \
	MOVOU 32(B1), X2 \
	PAND X2, X1 \
	MOVOU 48(B1), X2 \
	PAND X2, X1 \
	PAND X1, X0 \
	PTEST yccAlphaFF<>(SB), X0

// CONV4_SSE converts P1 (X0) and P2 (X1) of 4 pixels to the 32-bit Y,
// Cb and Cr samples in YD, BD and RD (Cb, Cr not yet clamped). Clobbers
// X0-X2.
#define CONV4_SSE(YD, BD, RD) \
	MOVO X0, BD \
	PMADDWL yccCb1<>(SB), BD \
	MOVO X1, X2 \
	PMADDWL yccCb2<>(SB), X2 \
	PADDL X2, BD \
	PSRLL $15, BD \
	MOVO X0, RD \
	PMADDWL yccCr1<>(SB), RD \
	MOVO X1, X2 \
	PMADDWL yccCr2<>(SB), X2 \
	PADDL X2, RD \
	PSRLL $15, RD \
	MOVO X0, YD \
	PMADDWL yccY1<>(SB), YD \
	PMADDWL yccY2<>(SB), X1 \
	PADDL X0, YD \
	PADDL X1, YD \
	PSRLL $16, YD

// PACK8_SSE and PACK16_SSE narrow the 32-bit samples of 16 pixels to
// bytes: PACK8_SSE the Y, Cb, Cr of pixels 0-3 (X3-X5) and 4-7 (X6-X8)
// to words in X3-X5, PACK16_SSE then those of pixels 8-11 (X6-X8) and
// 12-15 (X9-X11), leaving the byte rows in X3, X4 and X5.
#define PACK8_SSE \
	PACKSSLW X6, X3 \
	PACKSSLW X7, X4 \
	PACKSSLW X8, X5

#define PACK16_SSE \
	PACKSSLW X9, X6 \
	PACKSSLW X10, X7 \
	PACKSSLW X11, X8 \
	PACKUSWB X6, X3 \
	PACKUSWB X7, X4 \
	PACKUSWB X8, X5

// YCC16_SSE converts the 16 RGBA pixels at (AX) to Y, Cb and Cr bytes in
// X3, X4 and X5. Clobbers X0-X11.
#define YCC16_SSE(K256) \
	LOAD4_SSE(0(AX), K256) \
	CONV4_SSE(X3, X4, X5) \
	LOAD4_SSE(16(AX), K256) \
	CONV4_SSE(X6, X7, X8) \
	PACK8_SSE \
	LOAD4_SSE(32(AX), K256) \
	CONV4_SSE(X6, X7, X8) \
	LOAD4_SSE(48(AX), K256) \
	CONV4_SSE(X9, X10, X11) \
	PACK16_SSE

// YCC16N_SSE is YCC16_SSE for NRGBA pixels.
#define YCC16N_SSE(K256, KA) \
	LOAD4N_SSE(0(AX), K256, KA) \
	CONV4_SSE(X3, X4, X5) \
	LOAD4N_SSE(16(AX), K256, KA) \
	CONV4_SSE(X6, X7, X8) \
	PACK8_SSE \
	LOAD4N_SSE(32(AX), K256, KA) \
	CONV4_SSE(X6, X7, X8) \
	LOAD4N_SSE(48(AX), K256, KA) \
	CONV4_SSE(X9, X10, X11) \
	PACK16_SSE

// func rgbaToYCbCrSSE41(rgba, y, cb, cr *byte, n int)
TEXT ·rgbaToYCbCrSSE41(SB), NOSPLIT, $0-40
	MOVQ rgba+0(FP), SI
	MOVQ y+8(FP), DI
	MOVQ cb+16(FP), R8
	MOVQ cr+24(FP), R9
	MOVQ n+32(FP), R10
	SUBQ $16, R10 // last block
	XORQ CX, CX   // block start i
	MOVOU yccShufRG<>(SB), X15
	MOVOU yccShufB<>(SB), X14
	MOVOU yccK256<>(SB), X13

rgbaloop:
	CMPQ    CX, R10
	CMOVQGT R10, CX // i = min(i, n-16)
	LEAQ    (SI)(CX*4), AX
	YCC16_SSE(X13)
	MOVOU   X3, (DI)(CX*1)
	MOVOU   X4, (R8)(CX*1)
	MOVOU   X5, (R9)(CX*1)
	CMPQ    CX, R10
	JEQ     rgbadone
	ADDQ    $16, CX
	JMP     rgbaloop

rgbadone:
	RET

// func nrgbaToYCbCrSSE41(nrgba, y, cb, cr *byte, n int)
TEXT ·nrgbaToYCbCrSSE41(SB), NOSPLIT, $0-40
	MOVQ nrgba+0(FP), SI
	MOVQ y+8(FP), DI
	MOVQ cb+16(FP), R8
	MOVQ cr+24(FP), R9
	MOVQ n+32(FP), R10
	SUBQ $16, R10
	XORQ CX, CX
	MOVOU yccShufRG<>(SB), X15
	MOVOU yccShufB<>(SB), X14
	MOVOU yccK256<>(SB), X13
	MOVOU yccShufA<>(SB), X12

nrgbaloop:
	CMPQ    CX, R10
	CMOVQGT R10, CX
	LEAQ    (SI)(CX*4), AX
	OPAQUE16_SSE(AX)
	JCS     nrgbaopaque
	YCC16N_SSE(X13, X12)
	JMP     nrgbastore

nrgbaopaque:
	YCC16_SSE(X13)

nrgbastore:
	MOVOU X3, (DI)(CX*1)
	MOVOU X4, (R8)(CX*1)
	MOVOU X5, (R9)(CX*1)
	CMPQ  CX, R10
	JEQ   nrgbadone
	ADDQ  $16, CX
	JMP   nrgbaloop

nrgbadone:
	RET

// func downsample2x2SSE41(r0, r1, out *byte, n int)
TEXT ·downsample2x2SSE41(SB), NOSPLIT, $0-32
	MOVQ r0+0(FP), SI
	MOVQ r1+8(FP), DI
	MOVQ out+16(FP), R8
	MOVQ n+24(FP), R10
	SUBQ $16, R10
	XORQ CX, CX
	MOVOU yccOnes<>(SB), X15
	MOVOU yccK8192<>(SB), X14

dsloop:
	CMPQ      CX, R10
	CMOVQGT   R10, CX
	MOVOU     (SI)(CX*2), X0
	MOVOU     16(SI)(CX*2), X1
	MOVOU     (DI)(CX*2), X2
	MOVOU     16(DI)(CX*2), X3
	PMADDUBSW X15, X0
	PMADDUBSW X15, X1
	PMADDUBSW X15, X2
	PMADDUBSW X15, X3
	PADDW     X2, X0
	PADDW     X3, X1
	PMULHRSW  X14, X0
	PMULHRSW  X14, X1
	PACKUSWB  X1, X0
	MOVOU     X0, (R8)(CX*1)
	CMPQ      CX, R10
	JEQ       dsdone
	ADDQ      $16, CX
	JMP       dsloop

dsdone:
	RET

// The 4:2:0 loops convert the 16-pixel blocks at pixel CX of both rows:
// SI/DX the input rows, DI/R8 the luma rows, R9/R10 the chroma rows, R11
// the last block (n-16). The chroma pair sums of the first row are kept
// in X12 (Cb) and X13 (Cr).
#define CHROMA0_SSE \
	PMADDUBSW yccOnes<>(SB), X4 \
	PMADDUBSW yccOnes<>(SB), X5 \
	MOVO X4, X12 \
	MOVO X5, X13

#define CHROMA1_SSE \
	PMADDUBSW yccOnes<>(SB), X4 \
	PMADDUBSW yccOnes<>(SB), X5 \
	PADDW X12, X4 \
	PADDW X13, X5 \
	PMULHRSW yccK8192<>(SB), X4 \
	PMULHRSW yccK8192<>(SB), X5 \
	PACKUSWB X5, X4 \
	MOVQ CX, BX \
	SHRQ $1, BX \
	MOVQ X4, (R9)(BX*1) \
	MOVHPS X4, (R10)(BX*1)

// func rgbaToYCbCr420SSE41(rgba0, rgba1, y0, y1, cb, cr *byte, n int)
TEXT ·rgbaToYCbCr420SSE41(SB), NOSPLIT, $0-56
	MOVQ rgba0+0(FP), SI
	MOVQ rgba1+8(FP), DX
	MOVQ y0+16(FP), DI
	MOVQ y1+24(FP), R8
	MOVQ cb+32(FP), R9
	MOVQ cr+40(FP), R10
	MOVQ n+48(FP), R11
	SUBQ $16, R11
	XORQ CX, CX
	MOVOU yccShufRG<>(SB), X15
	MOVOU yccShufB<>(SB), X14

r420loop:
	CMPQ    CX, R11
	CMOVQGT R11, CX
	LEAQ    (SI)(CX*4), AX
	YCC16_SSE(yccK256<>(SB))
	MOVOU   X3, (DI)(CX*1)
	CHROMA0_SSE
	LEAQ    (DX)(CX*4), AX
	YCC16_SSE(yccK256<>(SB))
	MOVOU   X3, (R8)(CX*1)
	CHROMA1_SSE
	CMPQ    CX, R11
	JEQ     r420done
	ADDQ    $16, CX
	JMP     r420loop

r420done:
	RET

// func nrgbaToYCbCr420SSE41(nrgba0, nrgba1, y0, y1, cb, cr *byte, n int)
TEXT ·nrgbaToYCbCr420SSE41(SB), NOSPLIT, $0-56
	MOVQ nrgba0+0(FP), SI
	MOVQ nrgba1+8(FP), DX
	MOVQ y0+16(FP), DI
	MOVQ y1+24(FP), R8
	MOVQ cb+32(FP), R9
	MOVQ cr+40(FP), R10
	MOVQ n+48(FP), R11
	SUBQ $16, R11
	XORQ CX, CX
	MOVOU yccShufRG<>(SB), X15
	MOVOU yccShufB<>(SB), X14

n420loop:
	CMPQ    CX, R11
	CMOVQGT R11, CX
	LEAQ    (SI)(CX*4), AX
	LEAQ    (DX)(CX*4), R12
	OPAQUE16x2_SSE(AX, R12)
	JCS     n420opaque
	YCC16N_SSE(yccK256<>(SB), yccShufA<>(SB))
	MOVOU   X3, (DI)(CX*1)
	CHROMA0_SSE
	MOVQ    R12, AX
	YCC16N_SSE(yccK256<>(SB), yccShufA<>(SB))
	MOVOU   X3, (R8)(CX*1)
	CHROMA1_SSE
	JMP     n420next

n420opaque:
	YCC16_SSE(yccK256<>(SB))
	MOVOU X3, (DI)(CX*1)
	CHROMA0_SSE
	MOVQ  R12, AX
	YCC16_SSE(yccK256<>(SB))
	MOVOU X3, (R8)(CX*1)
	CHROMA1_SSE

n420next:
	CMPQ CX, R11
	JEQ  n420done
	ADDQ $16, CX
	JMP  n420loop

n420done:
	RET

// ---- AVX2 -----------------------------------------------------------------

// LOAD8_AVX loads the 8 RGBA pixels at OFF(AX) as P1 (Y1) and P2 (Y0).
// Y15 holds yccShufRG; KB is yccShufB (register or memory).
#define LOAD8_AVX(OFF, KB) \
	VMOVDQU OFF(AX), Y0 \
	VPSHUFB Y15, Y0, Y1 \
	VPSHUFB KB, Y0, Y0 \
	VPOR yccK256<>(SB), Y0, Y0

// LOAD8N_AVX is LOAD8_AVX for NRGBA pixels: it premultiplies P1 and P2.
// Clobbers Y2.
#define LOAD8N_AVX(OFF, KB) \
	VMOVDQU OFF(AX), Y0 \
	VPSHUFB yccShufA<>(SB), Y0, Y2 \
	VPSHUFB Y15, Y0, Y1 \
	VPSHUFB KB, Y0, Y0 \
	VPMULLW Y2, Y1, Y1 \
	VPMULLW Y2, Y0, Y0 \
	VPMULHUW yccK33025<>(SB), Y1, Y1 \
	VPMULHUW yccK33025<>(SB), Y0, Y0 \
	VPSRLW $7, Y1, Y1 \
	VPSRLW $7, Y0, Y0 \
	VPOR yccK256<>(SB), Y0, Y0

// OPAQUE32_AVX sets CF if the 32 NRGBA pixels at (BASE) all have alpha
// 255. Clobbers Y0.
#define OPAQUE32_AVX(BASE) \
	VMOVDQU (BASE), Y0 \
	VPAND 32(BASE), Y0, Y0 \
	VPAND 64(BASE), Y0, Y0 \
	VPAND 96(BASE), Y0, Y0 \
	VPTEST yccAlphaFF<>(SB), Y0

// OPAQUE32x2_AVX is OPAQUE32_AVX for the blocks at (B0) and (B1).
// Clobbers Y0, Y1.
#define OPAQUE32x2_AVX(B0, B1) \
	VMOVDQU (B0), Y0 \
	VPAND 32(B0), Y0, Y0 \
	VPAND 64(B0), Y0, Y0 \
	VPAND 96(B0), Y0, Y0 \
	VMOVDQU (B1), Y1 \
	VPAND 32(B1), Y1, Y1 \
	VPAND 64(B1), Y1, Y1 \
	VPAND 96(B1), Y1, Y1 \
	VPAND Y1, Y0, Y0 \
	VPTEST yccAlphaFF<>(SB), Y0

// CONV8_AVX converts P1 (Y1) and P2 (Y0) of 8 pixels to the 32-bit Y, Cb
// and Cr samples in YD, BD and RD. Clobbers Y0-Y2.
#define CONV8_AVX(YD, BD, RD) \
	VPMADDWD yccCb1<>(SB), Y1, BD \
	VPMADDWD yccCb2<>(SB), Y0, Y2 \
	VPADDD Y2, BD, BD \
	VPSRLD $15, BD, BD \
	VPMADDWD yccCr1<>(SB), Y1, RD \
	VPMADDWD yccCr2<>(SB), Y0, Y2 \
	VPADDD Y2, RD, RD \
	VPSRLD $15, RD, RD \
	VPMADDWD yccY1<>(SB), Y1, YD \
	VPMADDWD yccY2<>(SB), Y0, Y0 \
	VPADDD Y1, YD, YD \
	VPADDD Y0, YD, YD \
	VPSRLD $16, YD, YD

// PACK16_AVX and PACK32_AVX narrow the 32-bit samples of 32 pixels to
// bytes, as PACK8_SSE and PACK16_SSE (pixels 0-7 in Y3-Y5 and 8-15 in
// Y6-Y8 packed first, then 16-23 in Y6-Y8 and 24-31 in Y9-Y11). The
// in-lane packs leave the 4-pixel groups in the order 0 2 4 6 1 3 5 7
// (see yccPermD).
#define PACK16_AVX \
	VPACKSSDW Y6, Y3, Y3 \
	VPACKSSDW Y7, Y4, Y4 \
	VPACKSSDW Y8, Y5, Y5

#define PACK32_AVX \
	VPACKSSDW Y9, Y6, Y6 \
	VPACKSSDW Y10, Y7, Y7 \
	VPACKSSDW Y11, Y8, Y8 \
	VPACKUSWB Y6, Y3, Y3 \
	VPACKUSWB Y7, Y4, Y4 \
	VPACKUSWB Y8, Y5, Y5

// YCC32_AVX converts the 32 RGBA pixels at (AX) to Y, Cb and Cr bytes in
// Y3, Y4 and Y5, in the group order of PACK32_AVX. Clobbers Y0-Y11.
#define YCC32_AVX(KB) \
	LOAD8_AVX(0, KB) \
	CONV8_AVX(Y3, Y4, Y5) \
	LOAD8_AVX(32, KB) \
	CONV8_AVX(Y6, Y7, Y8) \
	PACK16_AVX \
	LOAD8_AVX(64, KB) \
	CONV8_AVX(Y6, Y7, Y8) \
	LOAD8_AVX(96, KB) \
	CONV8_AVX(Y9, Y10, Y11) \
	PACK32_AVX

// YCC32N_AVX is YCC32_AVX for NRGBA pixels.
#define YCC32N_AVX(KB) \
	LOAD8N_AVX(0, KB) \
	CONV8_AVX(Y3, Y4, Y5) \
	LOAD8N_AVX(32, KB) \
	CONV8_AVX(Y6, Y7, Y8) \
	PACK16_AVX \
	LOAD8N_AVX(64, KB) \
	CONV8_AVX(Y6, Y7, Y8) \
	LOAD8N_AVX(96, KB) \
	CONV8_AVX(Y9, Y10, Y11) \
	PACK32_AVX

// STORE32_AVX puts the bytes of YCC32_AVX in order (Y13 holds yccPermD)
// and stores them at pixel CX of the rows DI, R8 and R9.
#define STORE32_AVX \
	VPERMD Y3, Y13, Y3 \
	VPERMD Y4, Y13, Y4 \
	VPERMD Y5, Y13, Y5 \
	VMOVDQU Y3, (DI)(CX*1) \
	VMOVDQU Y4, (R8)(CX*1) \
	VMOVDQU Y5, (R9)(CX*1)

// func rgbaToYCbCrAVX2(rgba, y, cb, cr *byte, n int)
TEXT ·rgbaToYCbCrAVX2(SB), NOSPLIT, $0-40
	MOVQ rgba+0(FP), SI
	MOVQ y+8(FP), DI
	MOVQ cb+16(FP), R8
	MOVQ cr+24(FP), R9
	MOVQ n+32(FP), R10
	SUBQ $32, R10
	XORQ CX, CX
	VMOVDQU yccShufRG<>(SB), Y15
	VMOVDQU yccShufB<>(SB), Y14
	VMOVDQU yccPermD<>(SB), Y13

rgbaloop:
	CMPQ    CX, R10
	CMOVQGT R10, CX
	LEAQ    (SI)(CX*4), AX
	YCC32_AVX(Y14)
	STORE32_AVX
	CMPQ    CX, R10
	JEQ     rgbadone
	ADDQ    $32, CX
	JMP     rgbaloop

rgbadone:
	VZEROUPPER
	RET

// func nrgbaToYCbCrAVX2(nrgba, y, cb, cr *byte, n int)
TEXT ·nrgbaToYCbCrAVX2(SB), NOSPLIT, $0-40
	MOVQ nrgba+0(FP), SI
	MOVQ y+8(FP), DI
	MOVQ cb+16(FP), R8
	MOVQ cr+24(FP), R9
	MOVQ n+32(FP), R10
	SUBQ $32, R10
	XORQ CX, CX
	VMOVDQU yccShufRG<>(SB), Y15
	VMOVDQU yccShufB<>(SB), Y14
	VMOVDQU yccPermD<>(SB), Y13

nrgbaloop:
	CMPQ    CX, R10
	CMOVQGT R10, CX
	LEAQ    (SI)(CX*4), AX
	OPAQUE32_AVX(AX)
	JCS     nrgbaopaque
	YCC32N_AVX(Y14)
	JMP     nrgbastore

nrgbaopaque:
	YCC32_AVX(Y14)

nrgbastore:
	STORE32_AVX
	CMPQ CX, R10
	JEQ  nrgbadone
	ADDQ $32, CX
	JMP  nrgbaloop

nrgbadone:
	VZEROUPPER
	RET

// func downsample2x2AVX2(r0, r1, out *byte, n int)
TEXT ·downsample2x2AVX2(SB), NOSPLIT, $0-32
	MOVQ r0+0(FP), SI
	MOVQ r1+8(FP), DI
	MOVQ out+16(FP), R8
	MOVQ n+24(FP), R10
	SUBQ $32, R10
	XORQ CX, CX
	VMOVDQU yccOnes<>(SB), Y15
	VMOVDQU yccK8192<>(SB), Y14

dsloop:
	CMPQ       CX, R10
	CMOVQGT    R10, CX
	VMOVDQU    (SI)(CX*2), Y0
	VMOVDQU    32(SI)(CX*2), Y1
	VMOVDQU    (DI)(CX*2), Y2
	VMOVDQU    32(DI)(CX*2), Y3
	VPMADDUBSW Y15, Y0, Y0
	VPMADDUBSW Y15, Y1, Y1
	VPMADDUBSW Y15, Y2, Y2
	VPMADDUBSW Y15, Y3, Y3
	VPADDW     Y2, Y0, Y0
	VPADDW     Y3, Y1, Y1
	VPMULHRSW  Y14, Y0, Y0
	VPMULHRSW  Y14, Y1, Y1
	VPACKUSWB  Y1, Y0, Y0
	VPERMQ     $0xD8, Y0, Y0
	VMOVDQU    Y0, (R8)(CX*1)
	CMPQ       CX, R10
	JEQ        dsdone
	ADDQ       $32, CX
	JMP        dsloop

dsdone:
	VZEROUPPER
	RET

// The 4:2:0 loops convert the 32-pixel blocks at pixel CX of both rows:
// SI/DX the input rows, DI/R8 the luma rows, R9/R10 the chroma rows, R11
// the last block (n-32). The chroma pair sums of the first row are kept
// in Y12 (Cb) and Y14 (Cr). The pair sums keep the group order of
// PACK32_AVX (each 32-bit lane holds the two chroma samples of a 4-pixel
// group); STORE420_AVX packs Cb and Cr together, VPERMQ makes the low lane
// Cb's and the high lane Cr's, and yccFix420 orders the groups.
#define CHROMA0_AVX \
	VPMADDUBSW yccOnes<>(SB), Y4, Y12 \
	VPMADDUBSW yccOnes<>(SB), Y5, Y14

#define CHROMA1_AVX \
	VPMADDUBSW yccOnes<>(SB), Y4, Y4 \
	VPMADDUBSW yccOnes<>(SB), Y5, Y5 \
	VPADDW Y12, Y4, Y4 \
	VPADDW Y14, Y5, Y5 \
	VPMULHRSW yccK8192<>(SB), Y4, Y4 \
	VPMULHRSW yccK8192<>(SB), Y5, Y5 \
	VPACKUSWB Y5, Y4, Y4 \
	VPERMQ $0xD8, Y4, Y4 \
	VPSHUFB yccFix420<>(SB), Y4, Y4 \
	MOVQ CX, BX \
	SHRQ $1, BX \
	VMOVDQU X4, (R9)(BX*1) \
	VEXTRACTI128 $1, Y4, (R10)(BX*1)

// func rgbaToYCbCr420AVX2(rgba0, rgba1, y0, y1, cb, cr *byte, n int)
TEXT ·rgbaToYCbCr420AVX2(SB), NOSPLIT, $0-56
	MOVQ rgba0+0(FP), SI
	MOVQ rgba1+8(FP), DX
	MOVQ y0+16(FP), DI
	MOVQ y1+24(FP), R8
	MOVQ cb+32(FP), R9
	MOVQ cr+40(FP), R10
	MOVQ n+48(FP), R11
	SUBQ $32, R11
	XORQ CX, CX
	VMOVDQU yccShufRG<>(SB), Y15
	VMOVDQU yccPermD<>(SB), Y13

r420loop:
	CMPQ    CX, R11
	CMOVQGT R11, CX
	LEAQ    (SI)(CX*4), AX
	YCC32_AVX(yccShufB<>(SB))
	VPERMD  Y3, Y13, Y3
	VMOVDQU Y3, (DI)(CX*1)
	CHROMA0_AVX
	LEAQ    (DX)(CX*4), AX
	YCC32_AVX(yccShufB<>(SB))
	VPERMD  Y3, Y13, Y3
	VMOVDQU Y3, (R8)(CX*1)
	CHROMA1_AVX
	CMPQ    CX, R11
	JEQ     r420done
	ADDQ    $32, CX
	JMP     r420loop

r420done:
	VZEROUPPER
	RET

// func nrgbaToYCbCr420AVX2(nrgba0, nrgba1, y0, y1, cb, cr *byte, n int)
TEXT ·nrgbaToYCbCr420AVX2(SB), NOSPLIT, $0-56
	MOVQ nrgba0+0(FP), SI
	MOVQ nrgba1+8(FP), DX
	MOVQ y0+16(FP), DI
	MOVQ y1+24(FP), R8
	MOVQ cb+32(FP), R9
	MOVQ cr+40(FP), R10
	MOVQ n+48(FP), R11
	SUBQ $32, R11
	XORQ CX, CX
	VMOVDQU yccShufRG<>(SB), Y15
	VMOVDQU yccPermD<>(SB), Y13

n420loop:
	CMPQ    CX, R11
	CMOVQGT R11, CX
	LEAQ    (SI)(CX*4), AX
	LEAQ    (DX)(CX*4), R12
	OPAQUE32x2_AVX(AX, R12)
	JCS     n420opaque
	YCC32N_AVX(yccShufB<>(SB))
	VPERMD  Y3, Y13, Y3
	VMOVDQU Y3, (DI)(CX*1)
	CHROMA0_AVX
	MOVQ    R12, AX
	YCC32N_AVX(yccShufB<>(SB))
	VPERMD  Y3, Y13, Y3
	VMOVDQU Y3, (R8)(CX*1)
	CHROMA1_AVX
	JMP     n420next

n420opaque:
	YCC32_AVX(yccShufB<>(SB))
	VPERMD  Y3, Y13, Y3
	VMOVDQU Y3, (DI)(CX*1)
	CHROMA0_AVX
	MOVQ    R12, AX
	YCC32_AVX(yccShufB<>(SB))
	VPERMD  Y3, Y13, Y3
	VMOVDQU Y3, (R8)(CX*1)
	CHROMA1_AVX

n420next:
	CMPQ CX, R11
	JEQ  n420done
	ADDQ $32, CX
	JMP  n420loop

n420done:
	VZEROUPPER
	RET

// ---- AVX-512 --------------------------------------------------------------

// The AVX512 kernels use EVEX encodings, zmm registers and, beyond the
// AVX512 level (F, BW, DQ, VL), AVX512 VNNI: VPDPWSSD is PMADDWD plus the
// accumulation in one instruction, exact like PMADDWD followed by PADDD
// (the sums never overflow, see above). Blocks are 64 pixels, converted
// 16 at a time; the functions end with VZEROUPPER.
//
// Narrowing: the luma of a pixel is byte 2 of its 32-bit sum (which is
// below 1<<24), the low byte of the high word: VPERMT2W gathers the high
// words of two registers and VPMOVWB truncates them to bytes. Chroma is
// shifted by 15 and narrowed with unsigned saturation, which clamps 256
// to 255: VPMOVUSDB (4:4:4) or VPACKUSDW and VPACKUSWB (4:2:0: the pair
// sums only need the two pixels of a pair adjacent, which the in-lane
// packs keep).

// The AVX512 scheme: P2 = (b, g) (one VPSHUFB, no OR) and the biases in
// the accumulators (VPDPWSSD into a copy of the bias):
//   Y  = (32768 + P1.(19595, 19235) + P2.(7471, 19235)) >> 16
//   Cb = (4210688 + P1.(-5528, -10856) + P2.(16384, 0)) >> 15
//   Cr = (4210688 + P1.(16384, -13720) + P2.(-2664, 0)) >> 15
// the same sums as above (38470 = 2*19235, 4210688 = 257<<14).
C64X2(yccShufBG<>, $0x8005800680018002, $0x800D800E8009800A)
C64(yccY1Z<>, $0x4B234C8B4B234C8B)
C64(yccY2Z<>, $0x4B231D2F4B231D2F)
C64(yccCb2Z<>, $0x0000400000004000)
C64(yccCr2Z<>, $0x0000F5980000F598)
C64(yccBiasY<>, $0x0000800000008000)
C64(yccBiasC<>, $0x0040400000404000)

// VPERMT2W indices: the high 16-bit words of the 32-bit lanes of two
// registers (words 0-15 from the first, 16-31 from the second).
DATA yccIdxYW<>+0(SB)/8, $0x0007000500030001
DATA yccIdxYW<>+8(SB)/8, $0x000F000D000B0009
DATA yccIdxYW<>+16(SB)/8, $0x0017001500130011
DATA yccIdxYW<>+24(SB)/8, $0x001F001D001B0019
DATA yccIdxYW<>+32(SB)/8, $0x0027002500230021
DATA yccIdxYW<>+40(SB)/8, $0x002F002D002B0029
DATA yccIdxYW<>+48(SB)/8, $0x0037003500330031
DATA yccIdxYW<>+56(SB)/8, $0x003F003D003B0039
GLOBL yccIdxYW<>(SB), RODATA|NOPTR, $64

// VPERMW indices for the 4:2:0 chroma of 64 pixels: after the packs,
// 128-bit lane j holds the Cb samples of the 4-pixel groups j, 4+j, 8+j,
// 12+j (two samples, one 16-bit unit, each) and then their Cr samples;
// this puts the 32 Cb samples in order in the low half and the 32 Cr
// samples in the high half.
DATA yccFix420W<>+0(SB)/8, $0x0018001000080000
DATA yccFix420W<>+8(SB)/8, $0x0019001100090001
DATA yccFix420W<>+16(SB)/8, $0x001A0012000A0002
DATA yccFix420W<>+24(SB)/8, $0x001B0013000B0003
DATA yccFix420W<>+32(SB)/8, $0x001C0014000C0004
DATA yccFix420W<>+40(SB)/8, $0x001D0015000D0005
DATA yccFix420W<>+48(SB)/8, $0x001E0016000E0006
DATA yccFix420W<>+56(SB)/8, $0x001F0017000F0007
GLOBL yccFix420W<>(SB), RODATA|NOPTR, $64

// Constants in registers: Z31 yccShufRG, Z30 yccShufBG, Z29 yccBiasY,
// Z28 yccY1Z, Z27 yccY2Z, Z26 yccCb1, Z25 yccCb2Z, Z24 yccCr1, Z23
// yccCr2Z, Z22 yccIdxYW, Z21 yccShufA, Z20 yccK33025, Z19 yccOnes, Z18
// yccK8192, Z17 yccFix420W, Z16 yccBiasC.
#define CONSTS_Z \
	VMOVDQU64 yccShufRG<>(SB), Z31 \
	VMOVDQU64 yccShufBG<>(SB), Z30 \
	VMOVDQU64 yccBiasY<>(SB), Z29 \
	VMOVDQU64 yccY1Z<>(SB), Z28 \
	VMOVDQU64 yccY2Z<>(SB), Z27 \
	VMOVDQU64 yccCb1<>(SB), Z26 \
	VMOVDQU64 yccCb2Z<>(SB), Z25 \
	VMOVDQU64 yccCr1<>(SB), Z24 \
	VMOVDQU64 yccCr2Z<>(SB), Z23 \
	VMOVDQU64 yccIdxYW<>(SB), Z22 \
	VMOVDQU64 yccBiasC<>(SB), Z16

#define CONSTSN_Z \
	VMOVDQU64 yccShufA<>(SB), Z21 \
	VMOVDQU64 yccK33025<>(SB), Z20

#define CONSTS420_Z \
	VMOVDQU64 yccOnes<>(SB), Z19 \
	VMOVDQU64 yccK8192<>(SB), Z18 \
	VMOVDQU64 yccFix420W<>(SB), Z17

// LOAD16_Z loads the 16 RGBA pixels at OFF(AX) as P1 (Z1) and P2 = (b, g)
// (Z0).
#define LOAD16_Z(OFF) \
	VMOVDQU64 OFF(AX), Z0 \
	VPSHUFB Z31, Z0, Z1 \
	VPSHUFB Z30, Z0, Z0

// LOAD16N_Z is LOAD16_Z for NRGBA pixels. Clobbers Z2.
#define LOAD16N_Z(OFF) \
	VMOVDQU64 OFF(AX), Z0 \
	VPSHUFB Z21, Z0, Z2 \
	VPSHUFB Z31, Z0, Z1 \
	VPSHUFB Z30, Z0, Z0 \
	VPMULLW Z2, Z1, Z1 \
	VPMULLW Z2, Z0, Z0 \
	VPMULHUW Z20, Z1, Z1 \
	VPMULHUW Z20, Z0, Z0 \
	VPSRLW $7, Z1, Z1 \
	VPSRLW $7, Z0, Z0

// OPAQUE64_Z sets CF if the 64 NRGBA pixels at (BASE) all have alpha 255.
// Clobbers Z0, K1.
#define OPAQUE64_Z(BASE) \
	VMOVDQU64 (BASE), Z0 \
	VPANDD 64(BASE), Z0, Z0 \
	VPANDD 128(BASE), Z0, Z0 \
	VPANDD 192(BASE), Z0, Z0 \
	VPCMPUD $5, yccAlphaFF<>(SB), Z0, K1 \
	KORTESTW K1, K1

// OPAQUE64x2_Z is OPAQUE64_Z for the blocks at (B0) and (B1). Clobbers
// Z0, Z1, K1.
#define OPAQUE64x2_Z(B0, B1) \
	VMOVDQU64 (B0), Z0 \
	VPANDD 64(B0), Z0, Z0 \
	VPANDD 128(B0), Z0, Z0 \
	VPANDD 192(B0), Z0, Z0 \
	VMOVDQU64 (B1), Z1 \
	VPANDD 64(B1), Z1, Z1 \
	VPANDD 128(B1), Z1, Z1 \
	VPANDD 192(B1), Z1, Z1 \
	VPANDD Z1, Z0, Z0 \
	VPCMPUD $5, yccAlphaFF<>(SB), Z0, K1 \
	KORTESTW K1, K1

// CONV16_Z converts P1 (Z1) and P2 (Z0) of 16 pixels: the 32-bit luma
// sums in YD (luma in byte 2), Cb and Cr (shifted, not clamped) in BD and
// RD. The register copies of the biases cost no execution (they are
// eliminated at rename).
#define CONV16_Z(YD, BD, RD) \
	VMOVDQA64 Z16, BD \
	VPDPWSSD Z26, Z1, BD \
	VPDPWSSD Z25, Z0, BD \
	VMOVDQA64 Z16, RD \
	VPDPWSSD Z24, Z1, RD \
	VPDPWSSD Z23, Z0, RD \
	VMOVDQA64 Z29, YD \
	VPDPWSSD Z28, Z1, YD \
	VPDPWSSD Z27, Z0, YD \
	VPSRLD $15, BD, BD \
	VPSRLD $15, RD, RD

// STOREC16_Z stores the Cb (Z2) and Cr (Z3) samples of the 16 pixels at
// pixel CX+OFF to the rows R8 and R9.
#define STOREC16_Z(OFF) \
	VPMOVUSDB Z2, OFF(R8)(CX*1) \
	VPMOVUSDB Z3, OFF(R9)(CX*1)

// STOREY64_Z stores the luma of the 64 pixels whose sums are in Z10-Z13
// at pixel CX of the row YR. Clobbers Z10, Z12.
#define STOREY64_Z(YR) \
	VPERMT2W Z11, Z22, Z10 \
	VPERMT2W Z13, Z22, Z12 \
	VPMOVWB Z10, (YR)(CX*1) \
	VPMOVWB Z12, 32(YR)(CX*1)

// YCC64_Z converts the 64 pixels at (AX) (LOAD is LOAD16_Z or LOAD16N_Z)
// and stores them at pixel CX of the rows DI, R8 and R9. Clobbers Z0-Z3,
// Z10-Z13.
#define YCC64_Z(LOAD) \
	LOAD(0) \
	CONV16_Z(Z10, Z2, Z3) \
	STOREC16_Z(0) \
	LOAD(64) \
	CONV16_Z(Z11, Z2, Z3) \
	STOREC16_Z(16) \
	LOAD(128) \
	CONV16_Z(Z12, Z2, Z3) \
	STOREC16_Z(32) \
	LOAD(192) \
	CONV16_Z(Z13, Z2, Z3) \
	STOREC16_Z(48) \
	STOREY64_Z(DI)

// func rgbaToYCbCrAVX512(rgba, y, cb, cr *byte, n int)
TEXT ·rgbaToYCbCrAVX512(SB), NOSPLIT, $0-40
	MOVQ rgba+0(FP), SI
	MOVQ y+8(FP), DI
	MOVQ cb+16(FP), R8
	MOVQ cr+24(FP), R9
	MOVQ n+32(FP), R10
	SUBQ $64, R10
	XORQ CX, CX
	CONSTS_Z

rgbaloop:
	CMPQ    CX, R10
	CMOVQGT R10, CX
	LEAQ    (SI)(CX*4), AX
	YCC64_Z(LOAD16_Z)
	CMPQ    CX, R10
	JEQ     rgbadone
	ADDQ    $64, CX
	JMP     rgbaloop

rgbadone:
	VZEROUPPER
	RET

// func nrgbaToYCbCrAVX512(nrgba, y, cb, cr *byte, n int)
TEXT ·nrgbaToYCbCrAVX512(SB), NOSPLIT, $0-40
	MOVQ nrgba+0(FP), SI
	MOVQ y+8(FP), DI
	MOVQ cb+16(FP), R8
	MOVQ cr+24(FP), R9
	MOVQ n+32(FP), R10
	SUBQ $64, R10
	XORQ CX, CX
	CONSTS_Z
	CONSTSN_Z

nrgbaloop:
	CMPQ    CX, R10
	CMOVQGT R10, CX
	LEAQ    (SI)(CX*4), AX
	OPAQUE64_Z(AX)
	JCS     nrgbaopaque
	YCC64_Z(LOAD16N_Z)
	JMP     nrgbanext

nrgbaopaque:
	YCC64_Z(LOAD16_Z)

nrgbanext:
	CMPQ CX, R10
	JEQ  nrgbadone
	ADDQ $64, CX
	JMP  nrgbaloop

nrgbadone:
	VZEROUPPER
	RET

// func downsample2x2AVX512(r0, r1, out *byte, n int)
TEXT ·downsample2x2AVX512(SB), NOSPLIT, $0-32
	MOVQ r0+0(FP), SI
	MOVQ r1+8(FP), DI
	MOVQ out+16(FP), R8
	MOVQ n+24(FP), R10
	SUBQ $64, R10
	XORQ CX, CX
	VMOVDQU64 yccOnes<>(SB), Z31
	VMOVDQU64 yccK8192<>(SB), Z30

dsloop:
	CMPQ       CX, R10
	CMOVQGT    R10, CX
	VMOVDQU64  (SI)(CX*2), Z0
	VMOVDQU64  64(SI)(CX*2), Z1
	VMOVDQU64  (DI)(CX*2), Z2
	VMOVDQU64  64(DI)(CX*2), Z3
	VPMADDUBSW Z31, Z0, Z0
	VPMADDUBSW Z31, Z1, Z1
	VPMADDUBSW Z31, Z2, Z2
	VPMADDUBSW Z31, Z3, Z3
	VPADDW     Z2, Z0, Z0
	VPADDW     Z3, Z1, Z1
	VPMULHRSW  Z30, Z0, Z0
	VPMULHRSW  Z30, Z1, Z1
	VPMOVWB    Z0, (R8)(CX*1)
	VPMOVWB    Z1, 32(R8)(CX*1)
	CMPQ       CX, R10
	JEQ        dsdone
	ADDQ       $64, CX
	JMP        dsloop

dsdone:
	VZEROUPPER
	RET

// ROW420_Z converts the 64 pixels at (AX) of one row (LOAD is LOAD16_Z or
// LOAD16N_Z), stores their luma at pixel CX of the row YR and leaves the
// pair sums of their Cb and Cr samples (16-bit, groups as in yccFix420W)
// in Z4 and Z5. Clobbers Z0-Z2, Z4-Z13.
#define ROW420_Z(LOAD, YR) \
	LOAD(0) \
	CONV16_Z(Z10, Z4, Z5) \
	LOAD(64) \
	CONV16_Z(Z11, Z6, Z7) \
	VPACKUSDW Z6, Z4, Z4 \
	VPACKUSDW Z7, Z5, Z5 \
	LOAD(128) \
	CONV16_Z(Z12, Z6, Z7) \
	LOAD(192) \
	CONV16_Z(Z13, Z8, Z9) \
	VPACKUSDW Z8, Z6, Z6 \
	VPACKUSDW Z9, Z7, Z7 \
	VPACKUSWB Z6, Z4, Z4 \
	VPACKUSWB Z7, Z5, Z5 \
	STOREY64_Z(YR) \
	VPMADDUBSW Z19, Z4, Z4 \
	VPMADDUBSW Z19, Z5, Z5

// STORE420_Z averages the pair sums of the first row (Z14, Z15) and the
// second (Z4, Z5) and stores the 32 Cb and Cr samples at sample CX/2 of
// the rows R9 and R10.
#define STORE420_Z \
	VPADDW Z14, Z4, Z4 \
	VPADDW Z15, Z5, Z5 \
	VPMULHRSW Z18, Z4, Z4 \
	VPMULHRSW Z18, Z5, Z5 \
	VPACKUSWB Z5, Z4, Z4 \
	VPERMW Z4, Z17, Z4 \
	MOVQ CX, BX \
	SHRQ $1, BX \
	VMOVDQU Y4, (R9)(BX*1) \
	VEXTRACTI64X4 $1, Z4, (R10)(BX*1)

// func rgbaToYCbCr420AVX512(rgba0, rgba1, y0, y1, cb, cr *byte, n int)
TEXT ·rgbaToYCbCr420AVX512(SB), NOSPLIT, $0-56
	MOVQ rgba0+0(FP), SI
	MOVQ rgba1+8(FP), DX
	MOVQ y0+16(FP), DI
	MOVQ y1+24(FP), R8
	MOVQ cb+32(FP), R9
	MOVQ cr+40(FP), R10
	MOVQ n+48(FP), R11
	SUBQ $64, R11
	XORQ CX, CX
	CONSTS_Z
	CONSTS420_Z

r420loop:
	CMPQ      CX, R11
	CMOVQGT   R11, CX
	LEAQ      (SI)(CX*4), AX
	ROW420_Z(LOAD16_Z, DI)
	VMOVDQA64 Z4, Z14
	VMOVDQA64 Z5, Z15
	LEAQ      (DX)(CX*4), AX
	ROW420_Z(LOAD16_Z, R8)
	STORE420_Z
	CMPQ      CX, R11
	JEQ       r420done
	ADDQ      $64, CX
	JMP       r420loop

r420done:
	VZEROUPPER
	RET

// func nrgbaToYCbCr420AVX512(nrgba0, nrgba1, y0, y1, cb, cr *byte, n int)
TEXT ·nrgbaToYCbCr420AVX512(SB), NOSPLIT, $0-56
	MOVQ nrgba0+0(FP), SI
	MOVQ nrgba1+8(FP), DX
	MOVQ y0+16(FP), DI
	MOVQ y1+24(FP), R8
	MOVQ cb+32(FP), R9
	MOVQ cr+40(FP), R10
	MOVQ n+48(FP), R11
	SUBQ $64, R11
	XORQ CX, CX
	CONSTS_Z
	CONSTSN_Z
	CONSTS420_Z

n420loop:
	CMPQ      CX, R11
	CMOVQGT   R11, CX
	LEAQ      (SI)(CX*4), AX
	LEAQ      (DX)(CX*4), R12
	OPAQUE64x2_Z(AX, R12)
	JCS       n420opaque
	ROW420_Z(LOAD16N_Z, DI)
	VMOVDQA64 Z4, Z14
	VMOVDQA64 Z5, Z15
	MOVQ      R12, AX
	ROW420_Z(LOAD16N_Z, R8)
	JMP       n420next

n420opaque:
	ROW420_Z(LOAD16_Z, DI)
	VMOVDQA64 Z4, Z14
	VMOVDQA64 Z5, Z15
	MOVQ      R12, AX
	ROW420_Z(LOAD16_Z, R8)

n420next:
	STORE420_Z
	CMPQ CX, R11
	JEQ  n420done
	ADDQ $64, CX
	JMP  n420loop

n420done:
	VZEROUPPER
	RET
