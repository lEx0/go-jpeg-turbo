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

import (
	"bufio"
	"io"
)

// Output handling. image/jpeg writes through a bufio.Writer of the default
// size (4096 bytes) when w is not already buffered, and otherwise calls
// w.Write for the header segments and w.WriteByte for every byte of the
// scan. The encoder collects its output in e.out instead and passes it on
// in a way that, seen from the writer, behaves the same:
//
//   - encOutChunked (w has no Flush+WriteByte): image/jpeg's bufio.Writer
//     only ever calls w.Write with a full 4096-byte buffer and, at the end,
//     with the rest, turning a short write into io.ErrShortWrite and
//     stopping at the first error. drain makes exactly those calls.
//   - encOutBufio (w is a *bufio.Writer, or a *bufio.ReadWriter whose
//     methods are its Writer's): the header goes through the same Write and
//     WriteByte calls as in image/jpeg, and the scan through Write calls no
//     longer than the free space of the buffer, which change its state, and
//     make it flush, exactly as one WriteByte per byte would.
//   - encOutWriter (any other writer): the header goes through the same
//     calls; the scan goes through Write calls of up to len(e.out) bytes
//     instead of one WriteByte per byte (the same bytes, and the first error
//     the writer reports; like image/jpeg, the count returned by Write is
//     not checked).
//
// The first error makes every later write a no-op, as in image/jpeg; the
// encoder still reads the remaining pixels when doing so is observable (see
// encStrip.pure).

type encOutMode uint8

const (
	encOutChunked encOutMode = iota
	encOutBufio
	encOutWriter
)

const (
	// encChunk is the size of image/jpeg's bufio.Writer.
	encChunk = 4096
	// encOutSize is the capacity of e.out. The scan loop drains e.out when
	// fewer than encOutReserve bytes are free, which is more than one MCU
	// can produce (see encMaxBlockBytes).
	encOutSize    = 64 << 10
	encOutReserve = 8 << 10
)

// encStart prepares e (fresh or from encPool) to encode to w.
func (e *encoder) encStart(w io.Writer) {
	e.err = nil
	e.w, e.bw, e.raw = nil, nil, nil
	if ww, ok := w.(writer); ok {
		e.mode = encOutWriter
		e.w = ww
		switch bw := ww.(type) {
		case *bufio.Writer:
			e.mode, e.bw = encOutBufio, bw
		case *bufio.ReadWriter:
			if bw.Writer != nil {
				e.mode, e.bw = encOutBufio, bw.Writer
			}
		}
	} else {
		e.mode = encOutChunked
		e.raw = w
	}
	if cap(e.out) < encOutSize {
		e.out = make([]byte, 0, encOutSize)
	}
	e.out = e.out[:0]
	e.bits, e.nBits = 0, 0
}

// encEnd drops the references e holds to the caller's objects before it
// goes back to encPool.
func (e *encoder) encEnd() {
	e.w, e.bw, e.raw = nil, nil, nil
	e.err = nil
	e.out = e.out[:0]
	e.encStrip.release()
}

// drain passes the bytes of e.out to the writer. Unless final, in the
// chunked mode it keeps the bytes after the last complete encChunk.
func (e *encoder) drain(final bool) {
	if e.err != nil {
		e.out = e.out[:0]
		return
	}
	p := e.out
	switch e.mode {
	case encOutChunked:
		// bufio.Writer.Flush.
		for len(p) >= encChunk || (final && len(p) > 0) {
			k := min(len(p), encChunk)
			n, err := e.raw.Write(p[:k])
			if n < k && err == nil {
				err = io.ErrShortWrite
			}
			if err != nil {
				e.err = err
				e.out = e.out[:0]
				return
			}
			p = p[k:]
		}
		e.out = e.out[:copy(e.out, p)]
		return
	case encOutBufio:
		for len(p) > 0 && e.err == nil {
			k := e.bw.Available()
			if k <= 0 {
				// The buffer is full: Write flushes it, then copies up
				// to Size bytes without another flush, as WriteByte
				// would.
				k = e.bw.Size()
			}
			if k <= 0 {
				// A zero bufio.Writer: do what image/jpeg does.
				e.err = e.bw.WriteByte(p[0])
				p = p[1:]
				continue
			}
			k = min(k, len(p))
			_, e.err = e.bw.Write(p[:k])
			p = p[k:]
		}
	case encOutWriter:
		if len(p) > 0 {
			_, e.err = e.w.Write(p)
		}
	}
	e.out = e.out[:0]
}
