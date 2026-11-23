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

package suite

import (
	"bytes"
	stdjpeg "image/jpeg"
	"reflect"
	"strconv"
	"testing"
)

// Check compares c with the standard library on every workload of the
// suite, as the jpeg package's tests do on many more inputs: the decoded
// images must be reflect.DeepEqual (plane padding included), DecodeConfig
// must return the same config and Encode must write the same bytes.
// run.sh runs it once per build before measuring, so that a broken or
// misconfigured build fails instead of being timed. With -short it leaves
// out the 12 MP workloads.
func Check(t *testing.T, c Codec) {
	for _, f := range DecodeFiles {
		t.Run("Decode/img="+f.Name, func(t *testing.T) {
			if f.File == "" && testing.Short() {
				t.Skip("the 12 MP photo in -short mode")
			}
			data, err := f.Read()
			if err != nil {
				t.Fatal(err)
			}
			want, err := stdjpeg.Decode(bytes.NewReader(data))
			if err != nil {
				t.Fatal(err)
			}
			got, err := c.Decode(bytes.NewReader(data))
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Errorf("Decode: the image (%T) differs from image/jpeg's (%T)", got, want)
			}
			wantCfg, err := stdjpeg.DecodeConfig(bytes.NewReader(data))
			if err != nil {
				t.Fatal(err)
			}
			gotCfg, err := c.DecodeConfig(bytes.NewReader(data))
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(gotCfg, wantCfg) {
				t.Errorf("DecodeConfig = %+v, image/jpeg gives %+v", gotCfg, wantCfg)
			}
		})
	}
	for _, name := range EncodeInputs {
		for _, q := range EncodeQualities {
			t.Run("Encode/img="+name+"/q="+strconv.Itoa(q), func(t *testing.T) {
				if name == Photo12RGBAName && testing.Short() {
					t.Skip("the 12 MP photo in -short mode")
				}
				m, err := EncodeSource(name)
				if err != nil {
					t.Fatal(err)
				}
				var want, got bytes.Buffer
				if err := stdjpeg.Encode(&want, m, &stdjpeg.Options{Quality: q}); err != nil {
					t.Fatal(err)
				}
				if err := c.Encode(&got, m, q); err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(got.Bytes(), want.Bytes()) {
					t.Errorf("Encode wrote %d bytes that differ from image/jpeg's %d", got.Len(), want.Len())
				}
			})
		}
	}
}
