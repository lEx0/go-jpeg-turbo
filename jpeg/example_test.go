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

package jpeg_test

// Examples for the package documentation. Their outputs are exact: the
// decoder and the encoder give the same results at every SIMD level and
// with -tags purego, so the examples check that too.

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"image/color"
	stdjpeg "image/jpeg"
	"io"
	"log"
	"os"
	"reflect"

	"github.com/lEx0/go-jpeg-turbo/jpeg"
)

// Code written for image/jpeg switches to this package by changing the
// import path, as here:
//
//	import "github.com/lEx0/go-jpeg-turbo/jpeg" // was: "image/jpeg"
func ExampleDecode() {
	f, err := os.Open("../testdata/video-001.q50.420.jpeg")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	img, err := jpeg.Decode(f)
	if err != nil {
		log.Fatal(err)
	}

	// As with image/jpeg, the concrete type depends on the file.
	switch m := img.(type) {
	case *image.YCbCr: // color; the planes keep the file's chroma subsampling
		fmt.Println("YCbCr", m.SubsampleRatio, m.Bounds())
	case *image.Gray: // one component
		fmt.Println("Gray", m.Bounds())
	case *image.RGBA: // RGB components, no color transform
		fmt.Println("RGBA", m.Bounds())
	case *image.CMYK: // Adobe CMYK or YCCK
		fmt.Println("CMYK", m.Bounds())
	}
	// At and the color models work as for any image.Image.
	fmt.Println(color.RGBAModel.Convert(img.At(75, 50)))
	// Output:
	// YCbCr YCbCrSubsampleRatio420 (0,0)-(150,103)
	// {160 79 0 255}
}

// DecodeConfig reads only the header, so it is cheap to check the
// dimensions of an untrusted file before Decode allocates its pixels.
func ExampleDecodeConfig() {
	data, err := os.ReadFile("../testdata/photo-960x1280.jpg")
	if err != nil {
		log.Fatal(err)
	}

	cfg, err := jpeg.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(cfg.Width, cfg.Height, cfg.ColorModel == color.YCbCrModel)

	const maxPixels = 50_000_000
	if int64(cfg.Width)*int64(cfg.Height) > maxPixels {
		fmt.Println("too large")
		return
	}
	img, err := jpeg.Decode(bytes.NewReader(data))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(img.Bounds())
	// Output:
	// 960 1280 true
	// (0,0)-(960,1280)
}

func ExampleEncode() {
	// A 256x128 gradient.
	img := image.NewRGBA(image.Rect(0, 0, 256, 128))
	for y := range 128 {
		for x := range 256 {
			img.SetRGBA(x, y, color.RGBA{R: uint8(x), G: uint8(2 * y), B: 128, A: 255})
		}
	}

	// The writer can be any io.Writer: a file, a network connection, an
	// http.ResponseWriter. A nil *Options means DefaultQuality.
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 90}); err != nil {
		log.Fatal(err)
	}
	fmt.Println(buf.Len(), "bytes")

	// Like image/jpeg, Encode writes baseline JPEG, 4:2:0 for color images.
	out, err := jpeg.Decode(&buf)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(out.Bounds(), out.(*image.YCbCr).SubsampleRatio)
	// Output:
	// 3374 bytes
	// (0,0)-(256,128) YCbCrSubsampleRatio420
}

// Importing the package registers the "jpeg" format with package image,
// as image/jpeg does, so image.Decode and image.DecodeConfig read JPEG
// files with it. A program that uses only those needs the import for its
// side effect alone:
//
//	import _ "github.com/lEx0/go-jpeg-turbo/jpeg" // was: _ "image/jpeg"
//
// If image/jpeg is linked into the program as well, both register "jpeg"
// and image.Decode uses the one registered first.
func Example_imageDecode() {
	f, err := os.Open("../testdata/video-005.gray.jpeg")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	cfg, format, err := image.DecodeConfig(f)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(format, cfg.Width, cfg.Height)

	if _, err := f.Seek(0, io.SeekStart); err != nil {
		log.Fatal(err)
	}
	img, format, err := image.Decode(f)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s %T %v\n", format, img, img.Bounds())
	// Output:
	// jpeg 150 103
	// jpeg *image.Gray (0,0)-(150,103)
}

// Decode returns the same image as image/jpeg's Decode, down to the
// padding of the planes past the image edge, and Encode writes the same
// bytes. stdjpeg is the standard library's image/jpeg here.
func Example_compatibility() {
	data, err := os.ReadFile("../testdata/video-001.q50.422.jpeg")
	if err != nil {
		log.Fatal(err)
	}
	img, err := jpeg.Decode(bytes.NewReader(data))
	if err != nil {
		log.Fatal(err)
	}
	want, err := stdjpeg.Decode(bytes.NewReader(data))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("same image:", reflect.DeepEqual(img, want))

	var got, wantBytes bytes.Buffer
	if err := jpeg.Encode(&got, img, &jpeg.Options{Quality: 85}); err != nil {
		log.Fatal(err)
	}
	if err := stdjpeg.Encode(&wantBytes, img, &stdjpeg.Options{Quality: 85}); err != nil {
		log.Fatal(err)
	}
	fmt.Println("same bytes:", bytes.Equal(got.Bytes(), wantBytes.Bytes()))
	// Output:
	// same image: true
	// same bytes: true
}

// Decode and DecodeConfig fail in the same cases and with the same
// messages as image/jpeg: a FormatError for data that is not valid JPEG
// (data cut short in the middle of the compressed pixels included), an
// UnsupportedError for valid JPEG that uses a feature image/jpeg does not
// implement, and io.ErrUnexpectedEOF for data that ends inside the header.
// FormatError and UnsupportedError are this package's own types, distinct
// from image/jpeg's.
func Example_errors() {
	photo, err := os.ReadFile("../testdata/video-001.jpeg")
	if err != nil {
		log.Fatal(err)
	}
	arith, err := os.ReadFile("../testdata/corpus/photo16x16_arith.jpeg")
	if err != nil {
		log.Fatal(err)
	}

	for _, data := range [][]byte{
		[]byte("GIF89a"),     // not a JPEG at all
		arith,                // arithmetic coding
		photo[:len(photo)/2], // cut short in the compressed pixels
		photo[:100],          // cut short in the header
	} {
		_, err := jpeg.Decode(bytes.NewReader(data))
		var formatErr jpeg.FormatError
		var unsupportedErr jpeg.UnsupportedError
		switch {
		case errors.As(err, &formatErr):
			fmt.Println("invalid:", string(formatErr))
		case errors.As(err, &unsupportedErr):
			fmt.Println("unsupported:", string(unsupportedErr))
		case errors.Is(err, io.ErrUnexpectedEOF):
			fmt.Println("truncated")
		default:
			fmt.Println(err)
		}
	}
	// Output:
	// invalid: missing SOI marker
	// unsupported: unknown marker
	// invalid: short Huffman data
	// truncated
}
