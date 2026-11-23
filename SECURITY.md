# Security policy

## Supported versions

The module is at v0. Security fixes go into the latest minor version (the
highest v0.N) and are released as a new patch version of it. Older minor
versions do not get fixes: update to the latest release.

A fix is always a new version. Published versions are never moved or
deleted, because the Go module proxy keeps them; an affected version can
be retracted in `go.mod`, which makes `go get` and `go list -m -u` warn
about it.

## Reporting a vulnerability

Report it privately through GitHub:
<https://github.com/lEx0/go-jpeg-turbo/security/advisories/new>. Please do
not open a public issue, pull request or discussion about it, and keep the
details private until the advisory is published.

A report helps most with:

- the version or commit, the Go version, and `GOOS`/`GOARCH`;
- on amd64, the CPU, or the kernel level that fails: the tests can force
  one with `JPEG_TURBO_LEVEL=go|sse41|avx2|avx512 go test ./jpeg`; and
  whether `-tags purego` behaves the same;
- the input, or the code that makes it, and what happens: the crash
  output, the time or memory used, or how the result differs from
  `image/jpeg`'s.

## Scope

In scope is the code that a program importing
`github.com/lEx0/go-jpeg-turbo/jpeg` builds (the `jpeg` package and the
packages under `internal/` that it uses), and in it:

- memory safety of the decoder and the encoder: reading or writing outside
  a buffer, on any input or image;
- panics, hangs, or unbounded memory or time on malicious input;
- out-of-bounds access in the assembly kernels (NEON on arm64; SSE4.1,
  AVX2 and AVX-512 on amd64), for any input or image the API accepts;
- much more memory or time than `image/jpeg` needs for the same input, or
  output that depends on memory outside the input or the image (a sign of
  an out-of-bounds read).

Also in scope: the repository's workflows (`.github/`) and its release
tags, through which code could reach the module's users.

Not in scope:

- other differences from `image/jpeg` (a different image, error or panic
  on the same input): they are compatibility bugs, to be reported as a
  public issue;
- behaviour that `image/jpeg` shows the same way on the same input: report
  it to the Go project (<https://go.dev/security/policy>). That includes
  memory that grows with the image size an input declares; a program that
  decodes untrusted files should check the size `DecodeConfig` returns
  before it calls `Decode`;
- the benchmark module (`benchmarks/`), the examples (`_examples/`) and
  the tools (`_tools/`), which programs that import the package do not
  build.

## What to expect

This is a one-person project, so there is no fixed timeline. Reports are
answered on their advisory. If one is confirmed, the fix is prepared
privately and released as a new version, and the advisory is published
with it, crediting the reporter unless they ask otherwise. If it is not a
vulnerability, the answer says why and the advisory is closed.
