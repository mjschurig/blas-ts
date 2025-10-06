# BLAS-TS

[![npm version](https://badge.fury.io/js/blas-ts.svg)](https://badge.fury.io/js/blas-ts)
[![Build Status](https://github.com/username/blas-ts/workflows/Build%20and%20Publish/badge.svg)](https://github.com/username/blas-ts/actions)

Pure TypeScript implementation of BLAS (Basic Linear Algebra Subprograms).

This package provides high-performance linear algebra operations implemented in pure TypeScript, following the reference FORTRAN BLAS implementations for accuracy and performance.

## Features

- 🚀 **Pure TypeScript** - No native dependencies, works everywhere
- 📊 **BLAS Compatible** - Follows reference FORTRAN implementations
- 🎯 **Type Safe** - Full TypeScript support with detailed types
- ⚡ **Optimized** - Loop unrolling and performance optimizations
- 🌐 **Universal** - Works in Node.js, browsers, and edge environments
- ✅ **Comprehensive** - Complete Level 1, Level 2, and Level 3 BLAS operations

## Installation

```bash
npm install blas-ts
```

## Usage

### Level 1 BLAS - Vector Operations

```typescript
import { daxpy, ddot, dnrm2, dscal, dcopy } from 'blas-ts';

// DAXPY: y = alpha * x + y
const x = [1, 2, 3, 4];
const y = [5, 6, 7, 8];
daxpy(4, 2.0, x, 1, y, 1);
// y is now [7, 10, 13, 16]

// DDOT: compute dot product
const dot = ddot(4, x, 1, y, 1);

// DNRM2: compute Euclidean norm
const norm = dnrm2(4, x, 1);

// DSCAL: scale a vector
dscal(4, 2.0, x, 1); // x = 2.0 * x

// DCOPY: copy a vector
dcopy(4, x, 1, y, 1); // y = x
```

### Level 2 BLAS - Matrix-Vector Operations

```typescript
import { dgemv, dger, BLASTranspose } from 'blas-ts';

// DGEMV: matrix-vector multiply y = alpha*A*x + beta*y
const A = [1, 2, 3, 4, 5, 6]; // 2x3 matrix in column-major order
const x = [1, 2, 3];
const y = [0, 0];
dgemv(BLASTranspose.NoTranspose, 2, 3, 1.0, A, 2, x, 1, 0.0, y, 1);

// DGER: rank-1 update A = alpha*x*y^T + A
const x2 = [1, 2];
const y2 = [3, 4, 5];
dger(2, 3, 1.0, x2, 1, y2, 1, A, 2);
```

### Level 3 BLAS - Matrix-Matrix Operations

```typescript
import { dgemm, dsymm, BLASTranspose, BLASUplo, BLASSide } from 'blas-ts';

// DGEMM: general matrix multiply C = alpha*A*B + beta*C
const A = [1, 2, 3, 4]; // 2x2 matrix
const B = [5, 6, 7, 8]; // 2x2 matrix
const C = [0, 0, 0, 0]; // 2x2 result matrix
dgemm(BLASTranspose.NoTranspose, BLASTranspose.NoTranspose,
      2, 2, 2, 1.0, A, 2, B, 2, 0.0, C, 2);

// DSYMM: symmetric matrix multiply
dsymm(BLASSide.Left, BLASUplo.Upper, 2, 2, 1.0, A, 2, B, 2, 0.0, C, 2);
```

### Using with different vector types

```typescript
// Works with regular arrays
const x1: number[] = [1, 2, 3];
const y1: number[] = [4, 5, 6];

// Works with Float64Array
const x2 = new Float64Array([1, 2, 3]);
const y2 = new Float64Array([4, 5, 6]);

// Works with Float32Array
const x3 = new Float32Array([1, 2, 3]);
const y3 = new Float32Array([4, 5, 6]);

daxpy(3, 2.0, x1, 1, y1, 1);
daxpy(3, 2.0, x2, 1, y2, 1);
daxpy(3, 2.0, x3, 1, y3, 1);
```

## API Reference

### Level 1 BLAS (Vector-Vector Operations)

All Level 1 functions support strided access via `incx` and `incy` parameters.

- **`daxpy(n, alpha, x, incx, y, incy)`** - Compute `y = alpha*x + y`
- **`dscal(n, alpha, x, incx)`** - Scale vector: `x = alpha*x`
- **`dcopy(n, x, incx, y, incy)`** - Copy vector: `y = x`
- **`dswap(n, x, incx, y, incy)`** - Swap vectors: `x <-> y`
- **`ddot(n, x, incx, y, incy)`** - Dot product: returns `x^T * y`
- **`dnrm2(n, x, incx)`** - Euclidean norm: returns `||x||_2`
- **`dasum(n, x, incx)`** - Sum of absolute values: returns `Σ|x_i|`
- **`idamax(n, x, incx)`** - Index of maximum absolute value
- **`drotg(a, b)`** - Generate Givens rotation
- **`drot(n, x, incx, y, incy, c, s)`** - Apply Givens rotation

### Level 2 BLAS (Matrix-Vector Operations)

Matrix storage uses column-major order (Fortran-style). The leading dimension `ldA` specifies the stride between columns.

- **`dgemv(trans, m, n, alpha, A, ldA, x, incx, beta, y, incy)`** - General matrix-vector multiply: `y = alpha*op(A)*x + beta*y`
- **`dsymv(uplo, n, alpha, A, ldA, x, incx, beta, y, incy)`** - Symmetric matrix-vector multiply
- **`dtrmv(uplo, trans, diag, n, A, ldA, x, incx)`** - Triangular matrix-vector multiply: `x = op(A)*x`
- **`dtrsv(uplo, trans, diag, n, A, ldA, x, incx)`** - Solve triangular system: `op(A)*x = b`
- **`dger(m, n, alpha, x, incx, y, incy, A, ldA)`** - Rank-1 update: `A = alpha*x*y^T + A`
- **`dsyr(uplo, n, alpha, x, incx, A, ldA)`** - Symmetric rank-1 update: `A = alpha*x*x^T + A`
- **`dsyr2(uplo, n, alpha, x, incx, y, incy, A, ldA)`** - Symmetric rank-2 update: `A = alpha*x*y^T + alpha*y*x^T + A`

### Level 3 BLAS (Matrix-Matrix Operations)

All Level 3 operations use column-major matrix storage.

- **`dgemm(transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC)`** - General matrix multiply: `C = alpha*op(A)*op(B) + beta*C`
- **`dsymm(side, uplo, m, n, alpha, A, ldA, B, ldB, beta, C, ldC)`** - Symmetric matrix multiply
- **`dtrmm(side, uplo, transA, diag, m, n, alpha, A, ldA, B, ldB)`** - Triangular matrix multiply: `B = alpha*op(A)*B` or `B = alpha*B*op(A)`
- **`dtrsm(side, uplo, transA, diag, m, n, alpha, A, ldA, B, ldB)`** - Solve triangular system: `op(A)*X = alpha*B` or `X*op(A) = alpha*B`
- **`dsyrk(uplo, trans, n, k, alpha, A, ldA, beta, C, ldC)`** - Symmetric rank-k update: `C = alpha*A*A^T + beta*C` or `C = alpha*A^T*A + beta*C`
- **`dsyr2k(uplo, trans, n, k, alpha, A, ldA, B, ldB, beta, C, ldC)`** - Symmetric rank-2k update

### Type Enums

```typescript
enum BLASTranspose {
  NoTranspose,        // Use A
  Transpose,          // Use A^T
  ConjugateTranspose  // Use A^H (for complex matrices)
}

enum BLASUplo {
  Upper,  // Upper triangular
  Lower   // Lower triangular
}

enum BLASDiag {
  NonUnit,  // Diagonal is stored in matrix
  Unit      // Diagonal is assumed to be 1
}

enum BLASSide {
  Left,   // op(A)*B
  Right   // B*op(A)
}
```

## Development

This project uses a devcontainer for consistent development environment.

1. Open in VS Code with the Dev Containers extension
2. Rebuild and reopen in container when prompted
3. Start developing!

### Building

```bash
npm run build
```

### Development mode (watch)

```bash
npm run dev
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT

## Roadmap

### Completed ✅

- [x] **Level 1 BLAS** - Complete implementation with 10 functions (DAXPY, DSCAL, DCOPY, DSWAP, DDOT, DNRM2, DASUM, IDAMAX, DROTG, DROT)
- [x] **Level 2 BLAS** - Complete implementation with 7 functions (DGEMV, DSYMV, DTRMV, DTRSV, DGER, DSYR, DSYR2)
- [x] **Level 3 BLAS** - Complete implementation with 6 functions (DGEMM, DSYMM, DTRMM, DTRSM, DSYRK, DSYR2K)
- [x] **Comprehensive test suite** - 114+ tests covering all operations
- [x] **Loop unrolling optimizations** - Performance optimizations in Level 1 operations
- [x] **TypeScript types** - Full type safety with enums for BLAS parameters

### Future Enhancements 🚀

- [ ] Complex number support (ZGEMM, ZAXPY, etc.)
- [ ] Single precision support (SGEMM, SAXPY, etc.)
- [ ] Performance benchmarks
- [ ] SIMD optimizations (where available)
- [ ] Additional Level 1 operations (DSDOT, DROTM, DROTMG)
- [ ] Banded matrix operations (DGBMV, DSBMV, etc.)
- [ ] Packed storage format support (DSPMV, DSPR, etc.)
