// Benchmarks comparing this repository with the standard library's
// image/jpeg. A separate module, so that the library's module (what users
// download) carries neither the harness nor its generated inputs, and so
// that go test ./... of the library does not run it. See run.sh and the
// Benchmarks section of ../README.md.
module github.com/lEx0/go-jpeg-turbo/benchmarks

go 1.27

require github.com/lEx0/go-jpeg-turbo v0.1.0

require golang.org/x/sys v0.50.0 // indirect

replace github.com/lEx0/go-jpeg-turbo => ../
