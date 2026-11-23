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

package jpeg

import "github.com/lEx0/go-jpeg-turbo/internal/dct"

// decScanComp is a scan component selector of an SOS segment.
type decScanComp struct {
	compIndex uint8
	td        uint8 // DC table selector.
	ta        uint8 // AC table selector.
}

// decMinFastBlocks is the number of blocks below which a scan is decoded by
// the exact code alone: building the fast tables would cost more than it
// saves. (A variable so that tests can force the fast path.)
var decMinFastBlocks = 64

// decSetQuant updates the natural-order copy of quantization table tq.
func (d *decoder) decSetQuant(tq uint8) {
	q := &d.quant[tq]
	nq := &d.decQuant[tq]
	for zig, k := range decUnzig {
		nq[k] = q[zig]
	}
}

// decScanSeq decodes the entropy-coded data of a sequential (not
// progressive) scan: zigStart = 0, zigEnd = 63, ah = al = 0. It is
// processSOS's block loop, with the same block order, restart handling and
// errors, using the fast entropy decoder (decode_huffman.go) where it
// applies and reconstructing every block right away.
func (d *decoder) decScanSeq(nComp int, scan *[maxComponents]decScanComp, mxx, myy int) error {
	type scanComp struct {
		compIndex int
		hi, vi    int
		dcH, acH  *huffman
		qt        *dct.Block
		pix       []byte
		stride    int
	}
	var comps [maxComponents]scanComp
	fast := true
	nBlocks := 0
	for i := 0; i < nComp; i++ {
		ci := int(scan[i].compIndex)
		c := &comps[i]
		c.compIndex = ci
		c.hi, c.vi = d.comp[ci].h, d.comp[ci].v
		c.dcH = &d.huff[dcTable][scan[i].td]
		c.acH = &d.huff[acTable][scan[i].ta]
		c.qt = &d.decQuant[d.comp[ci].tq]
		c.pix, c.stride, _ = d.decPlane(ci)
		if c.dcH.nCodes == 0 || c.acH.nCodes == 0 {
			// decodeHuffman fails on such a table; leave that to the exact path.
			fast = false
		}
		nBlocks += c.hi * c.vi
	}
	if nBlocks*mxx*myy < decMinFastBlocks {
		fast = false
	}
	if fast {
		for i := 0; i < nComp; i++ {
			c := &comps[i]
			if !c.dcH.decLUTValid {
				c.dcH.decBuildLUT(dcTable)
			}
			if !c.acH.decLUTValid {
				c.acH.decBuildLUT(acTable)
			}
		}
	}

	// d.bits is canonical (empty) here, so the fast reader can take over.
	var br decBitReader
	if fast && !br.load(d) {
		fast = false
	}

	var (
		// b is the decoded coefficients, in natural (not zig-zag) order.
		b  block
		dc [maxComponents]int32
		// nbx and nby are the location of the next block of a
		// non-interleaved scan (processSOS's blockCount%q and blockCount/q).
		nbx, nby int
	)
	mcu, expectedRST := 0, uint8(rst0Marker)
	for my := 0; my < myy; my++ {
		for mx := 0; mx < mxx; mx++ {
			for i := 0; i < nComp; i++ {
				c := &comps[i]
				// See decScanProg for the block order. Non-interleaved scans
				// visit the blocks of their component in raster order and
				// skip those outside the image (leaving them zero).
				for v := 0; v < c.vi; v++ {
					for h := 0; h < c.hi; h++ {
						bx, by := c.hi*mx+h, c.vi*my+v
						if nComp == 1 {
							bx, by = nbx, nby
							if nbx++; nbx == mxx*c.hi {
								nbx, nby = 0, nby+1
							}
							if bx*8 >= d.width || by*8 >= d.height {
								continue
							}
						}

						b = block{}
						if !fast || d.eobRun != 0 || !d.decBlockSeq(&br, &b, c.dcH, c.acH, &dc[c.compIndex]) {
							if fast {
								// Redo the block with the exact code, from
								// the canonical state at its start.
								b = block{}
								br.store(d)
							}
							if err := d.decBlockExact(&b, c.dcH, c.acH, &dc[c.compIndex], 0, blockSize-1, 0, 0); err != nil {
								return err
							}
							if fast && !br.load(d) {
								fast = false
							}
						}

						if d.flex {
							if err := d.reconstructBlock(&b, bx, by, c.compIndex); err != nil {
								return err
							}
						} else {
							dct.DequantIDCTStore((*dct.Block)(&b), c.qt, c.pix[8*(by*c.stride+bx):], c.stride)
						}
					} // for h
				} // for v
			} // for i
			mcu++
			if d.ri > 0 && mcu%d.ri == 0 && mcu < mxx*myy {
				if fast {
					br.store(d)
				}
				if err := d.decProcessRST(&expectedRST); err != nil {
					return err
				}
				// Reset the DC components, as per section F.2.1.3.1.
				dc = [maxComponents]int32{}
				if fast && !br.load(d) {
					fast = false
				}
			}
		} // for mx
	} // for my
	if fast {
		br.store(d)
	}
	return nil
}

// Kinds of progressive scans, by their fast block decoder.
const (
	decDCFirst = iota
	decDCRefine
	decACFirst
	decACRefine
)

// decScanProg decodes the entropy-coded data of a progressive scan into
// d.progCoeffs. It is processSOS's block loop for progressive scans, with
// the same block order, restart handling and errors, using the fast entropy
// decoder where it applies.
//
// The blocks are traversed one MCU at a time. For 4:2:0 chroma subsampling,
// there are four Y 8x8 blocks in every 16x16 MCU. Interleaved scans (those
// with nComp > 1) visit the Y blocks of a 32x16 pixel image in this order:
//
//	0 1 4 5
//	2 3 6 7
//
// but non-interleaved scans are traversed left to right, top to bottom:
//
//	0 1 2 3
//	4 5 6 7
//
// Only DC scans (zigStart == 0) can be interleaved. AC scans must have only
// one component.
//
// To further complicate matters, for non-interleaved scans, there is no data
// for any blocks that are inside the image at the MCU level but outside the
// image at the pixel level. For example, a 24x16 pixel 4:2:0 progressive
// image consists of two 16x16 MCUs. The interleaved scans will process 8 Y
// blocks:
//
//	0 1 4 5
//	2 3 6 7
//
// The non-interleaved scans will process only 6 Y blocks:
//
//	0 1 2
//	3 4 5
//
// The blocks are decoded in place: image/jpeg copies each block out of
// d.progCoeffs and back, which only differs when decoding fails, and then
// the coefficients are never used.
func (d *decoder) decScanProg(nComp int, scan *[maxComponents]decScanComp, mxx, myy int, zigStart, zigEnd int32, ah, al uint32) error {
	type scanComp struct {
		compIndex int
		hi, vi    int
		dcH, acH  *huffman
		coeffs    []block
		nz        []uint64 // Non-zero masks (decoder.decNZ), if valid.
	}
	kind := decDCFirst
	switch {
	case zigStart == 0 && ah != 0:
		kind = decDCRefine
	case zigStart != 0 && ah == 0:
		kind = decACFirst
	case zigStart != 0:
		kind = decACRefine
	}
	var comps [maxComponents]scanComp
	fast := true
	nBlocks := 0
	for i := 0; i < nComp; i++ {
		ci := int(scan[i].compIndex)
		c := &comps[i]
		c.compIndex = ci
		c.hi, c.vi = d.comp[ci].h, d.comp[ci].v
		c.dcH = &d.huff[dcTable][scan[i].td]
		c.acH = &d.huff[acTable][scan[i].ta]
		c.coeffs = d.progCoeffs[ci]
		switch kind {
		case decDCFirst:
			if c.dcH.nCodes == 0 {
				fast = false
			}
		case decACFirst, decACRefine:
			if c.acH.nCodes == 0 {
				fast = false
			}
		}
		nBlocks += c.hi * c.vi
	}
	if nBlocks*mxx*myy < decMinFastBlocks {
		fast = false
	}
	if fast {
		for i := 0; i < nComp; i++ {
			c := &comps[i]
			if kind == decDCFirst && !c.dcH.decLUTValid {
				c.dcH.decBuildLUT(dcTable)
			}
			if (kind == decACFirst || kind == decACRefine) && !c.acH.decLUTValid {
				c.acH.decBuildLUT(acTable)
			}
		}
	}
	var br decBitReader
	if fast && !br.load(d) {
		fast = false
	}
	if kind == decACFirst || kind == decACRefine {
		// AC scans have one component. Its AC coefficients are all zero
		// before its first AC scan, and so are the masks.
		c := &comps[0]
		ci := c.compIndex
		if d.decNZ[ci] == nil {
			d.decNZ[ci] = make([]uint64, len(c.coeffs))
			d.decNZValid[ci] = true
		}
		if !fast {
			d.decNZValid[ci] = false
		}
		if d.decNZValid[ci] {
			c.nz = d.decNZ[ci]
		}
	}

	var (
		dc [maxComponents]int32
		// bx and by are the location of the current block, in units of 8x8
		// blocks: the third block in the first row has (bx, by) = (2, 0).
		bx, by     int
		blockCount int
	)
	zs, ze, sal := uint(zigStart), uint(zigEnd), uint(al)
	delta := int32(1) << al
	mcu, expectedRST := 0, uint8(rst0Marker)
	for my := 0; my < myy; my++ {
		for mx := 0; mx < mxx; mx++ {
			for i := 0; i < nComp; i++ {
				c := &comps[i]
				hi, vi := c.hi, c.vi
				for j := 0; j < hi*vi; j++ {
					if nComp != 1 {
						bx = hi*mx + j%hi
						by = vi*my + j/hi
					} else {
						q := mxx * hi
						bx = blockCount % q
						by = blockCount / q
						blockCount++
						if bx*8 >= d.width || by*8 >= d.height {
							continue
						}
					}
					k := by*mxx*hi + bx
					b := &c.coeffs[k]
					var nz *uint64
					if c.nz != nil {
						nz = &c.nz[k]
					}

					ok := false
					if fast {
						switch kind {
						case decDCFirst:
							ok = d.decBlockDCFirst(&br, b, c.dcH, &dc[c.compIndex], sal)
						case decDCRefine:
							ok = d.decBlockDCRefine(&br, b, delta)
						case decACFirst:
							if d.eobRun > 0 {
								d.eobRun--
								ok = true
							} else {
								ok = d.decBlockACFirst(&br, b, c.acH, zs, ze, sal, nz)
							}
						case decACRefine:
							ok = d.decBlockACRefine(&br, b, c.acH, zs, ze, delta, nz)
						}
					}
					if !ok {
						if fast {
							br.store(d)
						}
						if err := d.decBlockExact(b, c.dcH, c.acH, &dc[c.compIndex], zigStart, zigEnd, ah, al); err != nil {
							return err
						}
						if nz != nil {
							*nz = decNonZero(b)
						}
						if fast && !br.load(d) {
							fast = false
							if c.nz != nil {
								d.decNZValid[c.compIndex] = false
								c.nz = nil
							}
						}
					}
					// At this point, we could call reconstructBlock to dequantize and perform the
					// inverse DCT, to save early stages of a progressive image to the *image.YCbCr
					// buffers (the whole point of progressive encoding), but in Go, the jpeg.Decode
					// function does not return until the entire image is decoded, so we "continue"
					// here to avoid wasted computation. Instead, reconstructBlock is called on each
					// accumulated block by the reconstructProgressiveImage method after all of the
					// SOS markers are processed.
				} // for j
			} // for i
			mcu++
			if d.ri > 0 && mcu%d.ri == 0 && mcu < mxx*myy {
				if fast {
					br.store(d)
				}
				if err := d.decProcessRST(&expectedRST); err != nil {
					return err
				}
				// Reset the DC components, as per section F.2.1.3.1.
				dc = [maxComponents]int32{}
				if fast && !br.load(d) {
					fast = false
				}
			}
		} // for mx
	} // for my
	if fast {
		br.store(d)
	}
	return nil
}
