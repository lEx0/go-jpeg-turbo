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

package registertest

// The "jpeg" format registration of package jpeg, as a program that
// switched from image/jpeg with a blank import uses it. The tests of
// package jpeg cannot check it: their binary links image/jpeg too (they
// compare with it), image/jpeg's init runs first, and image.Decode then
// finds image/jpeg's decoder whatever package jpeg registers. This test
// binary links package jpeg and not image/jpeg, so image.Decode can only
// reach package jpeg's decoder; if image/jpeg were linked after all, the
// error types below would be image/jpeg's and the test would fail.

import (
	"bytes"
	"errors"
	"image"
	"os"
	"reflect"
	"testing"

	"github.com/lEx0/go-jpeg-turbo/jpeg"
)

func readFile(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(name)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// TestRegisterDecode checks that image.Decode and image.DecodeConfig find
// this package's decoder under the name "jpeg" and give the results of
// jpeg.Decode and jpeg.DecodeConfig.
func TestRegisterDecode(t *testing.T) {
	for _, f := range []string{"video-001.jpeg", "video-001.progressive.jpeg", "video-005.gray.jpeg", "video-001.cmyk.jpeg"} {
		data := readFile(t, "../../../testdata/"+f)
		m, name, err := image.Decode(bytes.NewReader(data))
		if err != nil || name != "jpeg" {
			t.Fatalf("%s: image.Decode: format %q, error %v", f, name, err)
		}
		want, err := jpeg.Decode(bytes.NewReader(data))
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(m, want) {
			t.Errorf("%s: image.Decode differs from jpeg.Decode", f)
		}
		cfg, name, err := image.DecodeConfig(bytes.NewReader(data))
		if err != nil || name != "jpeg" {
			t.Fatalf("%s: image.DecodeConfig: format %q, error %v", f, name, err)
		}
		if wantCfg, _ := jpeg.DecodeConfig(bytes.NewReader(data)); cfg != wantCfg {
			t.Errorf("%s: image.DecodeConfig: %+v, want %+v", f, cfg, wantCfg)
		}
	}
}

// TestRegisterErrors checks that the errors image.Decode and
// image.DecodeConfig return for bad JPEG data are this package's, of the
// same type and text as those of jpeg.Decode and jpeg.DecodeConfig.
func TestRegisterErrors(t *testing.T) {
	for _, c := range []struct {
		name string
		data []byte
		want any // A pointer to the error type of Decode.
	}{
		// A DQT segment with a precision of 2: "bad Pq value" (DecodeConfig
		// skips the segment and reaches the end of the data).
		{"FormatError", []byte("\xff\xd8\xff\xdb\x00\x43\x20"), new(jpeg.FormatError)},
		// An SOF segment of length 2: "number of components".
		{"UnsupportedError", []byte("\xff\xd8\xff\xc0\x00\x02"), new(jpeg.UnsupportedError)},
	} {
		_, _, err := image.Decode(bytes.NewReader(c.data))
		if !errors.As(err, c.want) {
			t.Errorf("%s: image.Decode error %v is a %T, not this package's", c.name, err, err)
		}
		_, wantErr := jpeg.Decode(bytes.NewReader(c.data))
		sameError(t, c.name+": image.Decode", err, wantErr)
		_, _, err = image.DecodeConfig(bytes.NewReader(c.data))
		_, wantErr = jpeg.DecodeConfig(bytes.NewReader(c.data))
		sameError(t, c.name+": image.DecodeConfig", err, wantErr)
	}
}

// sameError checks that err has the text and the type of want.
func sameError(t *testing.T, name string, err, want error) {
	t.Helper()
	if err == nil || want == nil || err.Error() != want.Error() || reflect.TypeOf(err) != reflect.TypeOf(want) {
		t.Errorf("%s: error %v (%T), want %v (%T)", name, err, err, want, want)
	}
}
