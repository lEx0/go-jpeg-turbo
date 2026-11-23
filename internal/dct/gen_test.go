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
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// TestGenerated checks that dct_arm64.s and dct_amd64.s are what asmgen.go
// and asmgen_x86.go generate.
func TestGenerated(t *testing.T) {
	if testing.Short() {
		t.Skip("runs the generators")
	}
	goTool, err := exec.LookPath("go")
	if err != nil {
		t.Skip("no go command")
	}
	for _, f := range []struct{ gen, file string }{
		{"asmgen.go", "dct_arm64.s"},
		{"asmgen_x86.go", "dct_amd64.s"},
	} {
		out := filepath.Join(t.TempDir(), f.file)
		cmd := exec.Command(goTool, "run", f.gen, "-o", out)
		if msg, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("go run %s: %v\n%s", f.gen, err, msg)
		}
		got, err := os.ReadFile(out)
		if err != nil {
			t.Fatal(err)
		}
		want, err := os.ReadFile(f.file)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Errorf("%s is out of date: run go generate", f.file)
		}
	}
}
