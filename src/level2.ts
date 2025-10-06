import { BLASTranspose, Matrix, Vector, BLASUplo, BLASDiag } from "./types";

/**
 * DGEMV: performs y := alpha*A*x + beta*y or y := alpha*A^T*x + beta*y
 *
 * @param trans - transpose type
 * @param m - number of rows of matrix A
 * @param n - number of columns of matrix A
 * @param alpha - scalar multiplier
 * @param A - input matrix
 * @param ldA - leading dimension of A (must be >= max(1, m))
 * @param x - input vector
 * @param incx - increment for elements of x
 * @param beta - scalar multiplier
 * @param y - input/output vector
 * @param incy - increment for elements of y
 */
export function dgemv(
  trans: BLASTranspose,
  m: number,
  n: number,
  alpha: number,
  A: Matrix,
  ldA: number,
  x: Vector,
  incx: number,
  beta: number,
  y: Vector,
  incy: number
): void {
  // DGEMV: performs y := alpha*A*x + beta*y or y := alpha*A^T*x + beta*y

  // Input validation (following Fortran reference lines 196-214)
  if (ldA < Math.max(1, m)) {
    throw new Error(
      `Invalid ldA: ${ldA}. Must be at least max(1, m) = ${Math.max(1, m)}`
    );
  }

  // Quick return if possible
  if (m === 0 || n === 0 || (alpha === 0.0 && beta === 1.0)) return;

  // Set vector lengths and start points
  let lenx: number, leny: number;
  if (trans === BLASTranspose.NoTranspose) {
    lenx = n;
    leny = m;
  } else {
    lenx = m;
    leny = n;
  }

  const kx = incx > 0 ? 0 : -(lenx - 1) * incx;
  const ky = incy > 0 ? 0 : -(leny - 1) * incy;

  // First form y := beta*y
  if (beta !== 1.0) {
    if (incy === 1) {
      if (beta === 0.0) {
        for (let i = 0; i < leny; i++) {
          y[i] = 0.0;
        }
      } else {
        for (let i = 0; i < leny; i++) {
          y[i] = beta * y[i];
        }
      }
    } else {
      let iy = ky;
      if (beta === 0.0) {
        for (let i = 0; i < leny; i++) {
          y[iy] = 0.0;
          iy += incy;
        }
      } else {
        for (let i = 0; i < leny; i++) {
          y[iy] = beta * y[iy];
          iy += incy;
        }
      }
    }
  }

  if (alpha === 0.0) return;

  if (trans === BLASTranspose.NoTranspose) {
    // Form y := alpha*A*x + y
    let jx = kx;
    if (incy === 1) {
      for (let j = 0; j < n; j++) {
        const temp = alpha * x[jx];
        for (let i = 0; i < m; i++) {
          y[i] = y[i] + temp * A[i + j * ldA];
        }
        jx += incx;
      }
    } else {
      for (let j = 0; j < n; j++) {
        const temp = alpha * x[jx];
        let iy = ky;
        for (let i = 0; i < m; i++) {
          y[iy] = y[iy] + temp * A[i + j * ldA];
          iy += incy;
        }
        jx += incx;
      }
    }
  } else {
    // Form y := alpha*A^T*x + y
    let jy = ky;
    if (incx === 1) {
      for (let j = 0; j < n; j++) {
        let temp = 0.0;
        for (let i = 0; i < m; i++) {
          temp = temp + A[i + j * ldA] * x[i];
        }
        y[jy] = y[jy] + alpha * temp;
        jy += incy;
      }
    } else {
      for (let j = 0; j < n; j++) {
        let temp = 0.0;
        let ix = kx;
        for (let i = 0; i < m; i++) {
          temp = temp + A[i + j * ldA] * x[ix];
          ix += incx;
        }
        y[jy] = y[jy] + alpha * temp;
        jy += incy;
      }
    }
  }
}

/**
 * DSYMV: performs y := alpha*A*x + beta*y for symmetric matrix A
 *
 * @param uplo - upper or lower triangular part of A
 * @param alpha - scalar multiplier
 * @param A - input matrix
 * @param x - input vector
 * @param beta - scalar multiplier
 * @param y - input/output vector
 */
export function dsymv(
  uplo: BLASUplo,
  n: number,
  alpha: number,
  A: Matrix,
  ldA: number,
  x: Vector,
  incx: number,
  beta: number,
  y: Vector,
  incy: number
): void {
  // DSYMV: performs y := alpha*A*x + beta*y for symmetric matrix A

  // Input validation
  if (uplo !== BLASUplo.Upper && uplo !== BLASUplo.Lower) {
    throw new Error("DSYMV: Invalid UPLO parameter");
  }
  if (n < 0) throw new Error("DSYMV: N must be >= 0");
  if (ldA < Math.max(1, n)) throw new Error("DSYMV: LDA must be >= max(1,N)");
  if (incx === 0) throw new Error("DSYMV: INCX must not be zero");
  if (incy === 0) throw new Error("DSYMV: INCY must not be zero");

  // Quick return if possible
  if (n === 0 || (alpha === 0.0 && beta === 1.0)) return;

  // Set up start points in X and Y
  const kx = incx > 0 ? 0 : -(n - 1) * incx;
  const ky = incy > 0 ? 0 : -(n - 1) * incy;

  // First form y := beta*y
  if (beta !== 1.0) {
    if (incy === 1) {
      if (beta === 0.0) {
        for (let i = 0; i < n; i++) {
          y[i] = 0.0;
        }
      } else {
        for (let i = 0; i < n; i++) {
          y[i] = beta * y[i];
        }
      }
    } else {
      let iy = ky;
      if (beta === 0.0) {
        for (let i = 0; i < n; i++) {
          y[iy] = 0.0;
          iy += incy;
        }
      } else {
        for (let i = 0; i < n; i++) {
          y[iy] = beta * y[iy];
          iy += incy;
        }
      }
    }
  }

  if (alpha === 0.0) return;

  if (uplo === BLASUplo.Upper) {
    // Form y when A is stored in upper triangle
    if (incx === 1 && incy === 1) {
      for (let j = 0; j < n; j++) {
        const temp1 = alpha * x[j];
        let temp2 = 0.0;

        // Upper triangular part (i < j)
        for (let i = 0; i < j; i++) {
          y[i] = y[i] + temp1 * A[i + j * ldA];
          temp2 = temp2 + A[i + j * ldA] * x[i];
        }

        // Diagonal element
        y[j] = y[j] + temp1 * A[j + j * ldA] + alpha * temp2;
      }
    } else {
      let jx = kx;
      let jy = ky;
      for (let j = 0; j < n; j++) {
        const temp1 = alpha * x[jx];
        let temp2 = 0.0;
        let ix = kx;
        let iy = ky;

        // Upper triangular part (i < j)
        for (let i = 0; i < j; i++) {
          y[iy] = y[iy] + temp1 * A[i + j * ldA];
          temp2 = temp2 + A[i + j * ldA] * x[ix];
          ix += incx;
          iy += incy;
        }

        // Diagonal element
        y[jy] = y[jy] + temp1 * A[j + j * ldA] + alpha * temp2;
        jx += incx;
        jy += incy;
      }
    }
  } else {
    // Form y when A is stored in lower triangle
    if (incx === 1 && incy === 1) {
      for (let j = 0; j < n; j++) {
        const temp1 = alpha * x[j];
        let temp2 = 0.0;

        // Diagonal element
        y[j] = y[j] + temp1 * A[j + j * ldA];

        // Lower triangular part (i > j)
        for (let i = j + 1; i < n; i++) {
          y[i] = y[i] + temp1 * A[i + j * ldA];
          temp2 = temp2 + A[i + j * ldA] * x[i];
        }

        y[j] = y[j] + alpha * temp2;
      }
    } else {
      let jx = kx;
      let jy = ky;
      for (let j = 0; j < n; j++) {
        const temp1 = alpha * x[jx];
        let temp2 = 0.0;

        // Diagonal element
        y[jy] = y[jy] + temp1 * A[j + j * ldA];

        let ix = jx;
        let iy = jy;

        // Lower triangular part (i > j)
        for (let i = j + 1; i < n; i++) {
          ix += incx;
          iy += incy;
          y[iy] = y[iy] + temp1 * A[i + j * ldA];
          temp2 = temp2 + A[i + j * ldA] * x[ix];
        }

        y[jy] = y[jy] + alpha * temp2;
        jx += incx;
        jy += incy;
      }
    }
  }
}

/**
 * DTRMV: performs x := A*x or x := A^T*x for triangular matrix A
 *
 * @param uplo - upper or lower triangular part of A
 * @param trans - transpose type
 * @param diag - diagonal type
 * @param A - input matrix
 * @param x - input/output vector
 */
export function dtrmv(
  uplo: BLASUplo,
  trans: BLASTranspose,
  diag: BLASDiag,
  n: number,
  A: Matrix,
  ldA: number,
  x: Vector,
  incx: number
): void {
  // DTRMV: performs x := A*x or x := A^T*x where A is triangular

  // Input validation
  if (uplo !== BLASUplo.Upper && uplo !== BLASUplo.Lower) {
    throw new Error("DTRMV: Invalid UPLO parameter");
  }
  if (
    trans !== BLASTranspose.NoTranspose &&
    trans !== BLASTranspose.Transpose &&
    trans !== BLASTranspose.ConjugateTranspose
  ) {
    throw new Error("DTRMV: Invalid TRANS parameter");
  }
  if (diag !== BLASDiag.Unit && diag !== BLASDiag.NonUnit) {
    throw new Error("DTRMV: Invalid DIAG parameter");
  }
  if (n < 0) throw new Error("DTRMV: N must be >= 0");
  if (ldA < Math.max(1, n)) throw new Error("DTRMV: LDA must be >= max(1,N)");
  if (incx === 0) throw new Error("DTRMV: INCX must not be zero");

  // Quick return if possible
  if (n === 0) return;

  const nounit = diag === BLASDiag.NonUnit;

  // Set up start point in X if increment is not unity
  let kx = 0;
  if (incx <= 0) {
    kx = -(n - 1) * incx;
  } else if (incx !== 1) {
    kx = 0;
  }

  if (trans === BLASTranspose.NoTranspose) {
    // Form x := A*x
    if (uplo === BLASUplo.Upper) {
      if (incx === 1) {
        for (let j = 0; j < n; j++) {
          if (x[j] !== 0.0) {
            const temp = x[j];
            for (let i = 0; i < j; i++) {
              x[i] = x[i] + temp * A[i + j * ldA];
            }
            if (nounit) x[j] = x[j] * A[j + j * ldA];
          }
        }
      } else {
        let jx = kx;
        for (let j = 0; j < n; j++) {
          if (x[jx] !== 0.0) {
            const temp = x[jx];
            let ix = kx;
            for (let i = 0; i < j; i++) {
              x[ix] = x[ix] + temp * A[i + j * ldA];
              ix += incx;
            }
            if (nounit) x[jx] = x[jx] * A[j + j * ldA];
          }
          jx += incx;
        }
      }
    } else {
      if (incx === 1) {
        for (let j = n - 1; j >= 0; j--) {
          if (x[j] !== 0.0) {
            const temp = x[j];
            for (let i = n - 1; i > j; i--) {
              x[i] = x[i] + temp * A[i + j * ldA];
            }
            if (nounit) x[j] = x[j] * A[j + j * ldA];
          }
        }
      } else {
        kx = kx + (n - 1) * incx;
        let jx = kx;
        for (let j = n - 1; j >= 0; j--) {
          if (x[jx] !== 0.0) {
            const temp = x[jx];
            let ix = kx;
            for (let i = n - 1; i > j; i--) {
              x[ix] = x[ix] + temp * A[i + j * ldA];
              ix -= incx;
            }
            if (nounit) x[jx] = x[jx] * A[j + j * ldA];
          }
          jx -= incx;
        }
      }
    }
  } else {
    // Form x := A^T*x
    if (uplo === BLASUplo.Upper) {
      if (incx === 1) {
        for (let j = n - 1; j >= 0; j--) {
          let temp = x[j];
          if (nounit) temp = temp * A[j + j * ldA];
          for (let i = j - 1; i >= 0; i--) {
            temp = temp + A[i + j * ldA] * x[i];
          }
          x[j] = temp;
        }
      } else {
        let jx = kx + (n - 1) * incx;
        for (let j = n - 1; j >= 0; j--) {
          let temp = x[jx];
          let ix = jx;
          if (nounit) temp = temp * A[j + j * ldA];
          for (let i = j - 1; i >= 0; i--) {
            ix -= incx;
            temp = temp + A[i + j * ldA] * x[ix];
          }
          x[jx] = temp;
          jx -= incx;
        }
      }
    } else {
      if (incx === 1) {
        for (let j = 0; j < n; j++) {
          let temp = x[j];
          if (nounit) temp = temp * A[j + j * ldA];
          for (let i = j + 1; i < n; i++) {
            temp = temp + A[i + j * ldA] * x[i];
          }
          x[j] = temp;
        }
      } else {
        let jx = kx;
        for (let j = 0; j < n; j++) {
          let temp = x[jx];
          let ix = jx;
          if (nounit) temp = temp * A[j + j * ldA];
          for (let i = j + 1; i < n; i++) {
            ix += incx;
            temp = temp + A[i + j * ldA] * x[ix];
          }
          x[jx] = temp;
          jx += incx;
        }
      }
    }
  }
}

/**
 * DTRSV: performs x := inv(A)*x or x := inv(A^T)*x for triangular matrix A
 *
 * @param uplo - upper or lower triangular part of A
 * @param trans - transpose type
 * @param diag - diagonal type
 * @param A - input matrix
 * @param x - input/output vector
 */
export function dtrsv(
  uplo: BLASUplo,
  trans: BLASTranspose,
  diag: BLASDiag,
  n: number,
  A: Matrix,
  ldA: number,
  x: Vector,
  incx: number
): void {
  // ✅ IMPLEMENTED: Following ./packages/numeric/reference-implementation/BLAS/SRC/dtrsv.f
  // DTRSV: solves A*x = b or A^T*x = b where A is triangular

  // Input validation
  if (uplo !== BLASUplo.Upper && uplo !== BLASUplo.Lower) {
    throw new Error("DTRSV: Invalid UPLO parameter");
  }
  if (
    trans !== BLASTranspose.NoTranspose &&
    trans !== BLASTranspose.Transpose &&
    trans !== BLASTranspose.ConjugateTranspose
  ) {
    throw new Error("DTRSV: Invalid TRANS parameter");
  }
  if (diag !== BLASDiag.Unit && diag !== BLASDiag.NonUnit) {
    throw new Error("DTRSV: Invalid DIAG parameter");
  }
  if (n < 0) throw new Error("DTRSV: N must be >= 0");
  if (ldA < Math.max(1, n)) throw new Error("DTRSV: LDA must be >= max(1,N)");
  if (incx === 0) throw new Error("DTRSV: INCX must not be zero");

  // Quick return if possible
  if (n === 0) return;

  const nounit = diag === BLASDiag.NonUnit;

  // Set up start point in X if increment is not unity
  let kx = 0;
  if (incx <= 0) {
    kx = -(n - 1) * incx;
  } else if (incx !== 1) {
    kx = 0;
  }

  if (trans === BLASTranspose.NoTranspose) {
    // Form x := inv(A)*x
    if (uplo === BLASUplo.Upper) {
      if (incx === 1) {
        for (let j = n - 1; j >= 0; j--) {
          if (x[j] !== 0.0) {
            if (nounit) x[j] = x[j] / A[j + j * ldA];
            const temp = x[j];
            for (let i = j - 1; i >= 0; i--) {
              x[i] = x[i] - temp * A[i + j * ldA];
            }
          }
        }
      } else {
        let jx = kx + (n - 1) * incx;
        for (let j = n - 1; j >= 0; j--) {
          if (x[jx] !== 0.0) {
            if (nounit) x[jx] = x[jx] / A[j + j * ldA];
            const temp = x[jx];
            let ix = jx;
            for (let i = j - 1; i >= 0; i--) {
              ix -= incx;
              x[ix] = x[ix] - temp * A[i + j * ldA];
            }
          }
          jx -= incx;
        }
      }
    } else {
      if (incx === 1) {
        for (let j = 0; j < n; j++) {
          if (x[j] !== 0.0) {
            if (nounit) x[j] = x[j] / A[j + j * ldA];
            const temp = x[j];
            for (let i = j + 1; i < n; i++) {
              x[i] = x[i] - temp * A[i + j * ldA];
            }
          }
        }
      } else {
        let jx = kx;
        for (let j = 0; j < n; j++) {
          if (x[jx] !== 0.0) {
            if (nounit) x[jx] = x[jx] / A[j + j * ldA];
            const temp = x[jx];
            let ix = jx;
            for (let i = j + 1; i < n; i++) {
              ix += incx;
              x[ix] = x[ix] - temp * A[i + j * ldA];
            }
          }
          jx += incx;
        }
      }
    }
  } else {
    // Form x := inv(A^T)*x
    if (uplo === BLASUplo.Upper) {
      if (incx === 1) {
        for (let j = 0; j < n; j++) {
          let temp = x[j];
          for (let i = 0; i < j; i++) {
            temp = temp - A[i + j * ldA] * x[i];
          }
          if (nounit) temp = temp / A[j + j * ldA];
          x[j] = temp;
        }
      } else {
        let jx = kx;
        for (let j = 0; j < n; j++) {
          let temp = x[jx];
          let ix = kx;
          for (let i = 0; i < j; i++) {
            temp = temp - A[i + j * ldA] * x[ix];
            ix += incx;
          }
          if (nounit) temp = temp / A[j + j * ldA];
          x[jx] = temp;
          jx += incx;
        }
      }
    } else {
      if (incx === 1) {
        for (let j = n - 1; j >= 0; j--) {
          let temp = x[j];
          for (let i = n - 1; i > j; i--) {
            temp = temp - A[i + j * ldA] * x[i];
          }
          if (nounit) temp = temp / A[j + j * ldA];
          x[j] = temp;
        }
      } else {
        kx = kx + (n - 1) * incx;
        let jx = kx;
        for (let j = n - 1; j >= 0; j--) {
          let temp = x[jx];
          let ix = kx;
          for (let i = n - 1; i > j; i--) {
            temp = temp - A[i + j * ldA] * x[ix];
            ix -= incx;
          }
          if (nounit) temp = temp / A[j + j * ldA];
          x[jx] = temp;
          jx -= incx;
        }
      }
    }
  }
}

/**
 * DGER: performs A := alpha*x*y^T + A
 *
 * @param alpha - scalar multiplier
 * @param x - input vector
 * @param y - input vector
 * @param A - input/output matrix
 */
export function dger(
  m: number,
  n: number,
  alpha: number,
  x: Vector,
  incx: number,
  y: Vector,
  incy: number,
  A: Matrix,
  ldA: number
): void {
  // DGER: performs the rank 1 operation A := alpha*x*y^T + A

  // Input validation
  if (m < 0) throw new Error("DGER: M must be >= 0");
  if (n < 0) throw new Error("DGER: N must be >= 0");
  if (incx === 0) throw new Error("DGER: INCX must not be zero");
  if (incy === 0) throw new Error("DGER: INCY must not be zero");
  if (ldA < Math.max(1, m)) throw new Error("DGER: LDA must be >= max(1,M)");

  // Quick return if possible
  if (m === 0 || n === 0 || alpha === 0.0) return;

  // Start the operations. In this version the elements of A are
  // accessed sequentially with one pass through A.
  let jy = incy > 0 ? 0 : -(n - 1) * incy;

  if (incx === 1) {
    for (let j = 0; j < n; j++) {
      if (y[jy] !== 0.0) {
        const temp = alpha * y[jy];
        for (let i = 0; i < m; i++) {
          A[i + j * ldA] = A[i + j * ldA] + x[i] * temp;
        }
      }
      jy += incy;
    }
  } else {
    const kx = incx > 0 ? 0 : -(m - 1) * incx;
    for (let j = 0; j < n; j++) {
      if (y[jy] !== 0.0) {
        const temp = alpha * y[jy];
        let ix = kx;
        for (let i = 0; i < m; i++) {
          A[i + j * ldA] = A[i + j * ldA] + x[ix] * temp;
          ix += incx;
        }
      }
      jy += incy;
    }
  }
}

/**
 * DSYR: performs A := alpha*x*x^T + A for symmetric matrix A
 *
 * @param uplo - upper or lower triangular part of A
 * @param alpha - scalar multiplier
 * @param x - input vector
 * @param A - input/output matrix
 */
export function dsyr(
  uplo: BLASUplo,
  n: number,
  alpha: number,
  x: Vector,
  incx: number,
  A: Matrix,
  ldA: number
): void {
  // DSYR: performs the symmetric rank 1 operation A := alpha*x*x^T + A

  // Input validation
  if (uplo !== BLASUplo.Upper && uplo !== BLASUplo.Lower) {
    throw new Error("DSYR: Invalid UPLO parameter");
  }
  if (n < 0) throw new Error("DSYR: N must be >= 0");
  if (incx === 0) throw new Error("DSYR: INCX must not be zero");
  if (ldA < Math.max(1, n)) throw new Error("DSYR: LDA must be >= max(1,N)");

  // Quick return if possible
  if (n === 0 || alpha === 0.0) return;

  // Set the start point in X if the increment is not unity
  let kx = 0;
  if (incx <= 0) {
    kx = -(n - 1) * incx;
  } else if (incx !== 1) {
    kx = 0;
  }

  if (uplo === BLASUplo.Upper) {
    // Form A when A is stored in upper triangle
    if (incx === 1) {
      for (let j = 0; j < n; j++) {
        if (x[j] !== 0.0) {
          const temp = alpha * x[j];
          for (let i = 0; i <= j; i++) {
            A[i + j * ldA] = A[i + j * ldA] + x[i] * temp;
          }
        }
      }
    } else {
      let jx = kx;
      for (let j = 0; j < n; j++) {
        if (x[jx] !== 0.0) {
          const temp = alpha * x[jx];
          let ix = kx;
          for (let i = 0; i <= j; i++) {
            A[i + j * ldA] = A[i + j * ldA] + x[ix] * temp;
            ix += incx;
          }
        }
        jx += incx;
      }
    }
  } else {
    // Form A when A is stored in lower triangle
    if (incx === 1) {
      for (let j = 0; j < n; j++) {
        if (x[j] !== 0.0) {
          const temp = alpha * x[j];
          for (let i = j; i < n; i++) {
            A[i + j * ldA] = A[i + j * ldA] + x[i] * temp;
          }
        }
      }
    } else {
      let jx = kx;
      for (let j = 0; j < n; j++) {
        if (x[jx] !== 0.0) {
          const temp = alpha * x[jx];
          let ix = jx;
          for (let i = j; i < n; i++) {
            A[i + j * ldA] = A[i + j * ldA] + x[ix] * temp;
            ix += incx;
          }
        }
        jx += incx;
      }
    }
  }
}

/**
 * DSYR2: performs A := alpha*x*y^T + alpha*y*x^T + A for symmetric matrix A
 *
 * @param uplo - upper or lower triangular part of A
 * @param n - order of the matrix A
 * @param alpha - scalar multiplier
 * @param x - input vector
 * @param incx - increment for elements of x
 * @param y - input vector
 * @param incy - increment for elements of y
 * @param A - input/output matrix
 * @param ldA - leading dimension of A
 */
export function dsyr2(
  uplo: BLASUplo,
  n: number,
  alpha: number,
  x: Vector,
  incx: number,
  y: Vector,
  incy: number,
  A: Matrix,
  ldA: number
): void {
  // DSYR2: performs the symmetric rank 2 operation A := alpha*x*y^T + alpha*y*x^T + A

  // Input validation
  if (uplo !== BLASUplo.Upper && uplo !== BLASUplo.Lower) {
    throw new Error("DSYR2: Invalid UPLO parameter");
  }
  if (n < 0) throw new Error("DSYR2: N must be >= 0");
  if (incx === 0) throw new Error("DSYR2: INCX must not be zero");
  if (incy === 0) throw new Error("DSYR2: INCY must not be zero");
  if (ldA < Math.max(1, n)) throw new Error("DSYR2: LDA must be >= max(1,N)");

  // Quick return if possible
  if (n === 0 || alpha === 0.0) return;

  // Set up start points in X and Y if increments are not both unity
  let kx = 0,
    ky = 0;
  if (incx !== 1 || incy !== 1) {
    kx = incx > 0 ? 0 : -(n - 1) * incx;
    ky = incy > 0 ? 0 : -(n - 1) * incy;
  }

  if (uplo === BLASUplo.Upper) {
    // Form A when A is stored in the upper triangle
    if (incx === 1 && incy === 1) {
      for (let j = 0; j < n; j++) {
        if (x[j] !== 0.0 || y[j] !== 0.0) {
          const temp1 = alpha * y[j];
          const temp2 = alpha * x[j];
          for (let i = 0; i <= j; i++) {
            A[i + j * ldA] = A[i + j * ldA] + x[i] * temp1 + y[i] * temp2;
          }
        }
      }
    } else {
      let jx = kx;
      let jy = ky;
      for (let j = 0; j < n; j++) {
        if (x[jx] !== 0.0 || y[jy] !== 0.0) {
          const temp1 = alpha * y[jy];
          const temp2 = alpha * x[jx];
          let ix = kx;
          let iy = ky;
          for (let i = 0; i <= j; i++) {
            A[i + j * ldA] = A[i + j * ldA] + x[ix] * temp1 + y[iy] * temp2;
            ix += incx;
            iy += incy;
          }
        }
        jx += incx;
        jy += incy;
      }
    }
  } else {
    // Form A when A is stored in the lower triangle
    if (incx === 1 && incy === 1) {
      for (let j = 0; j < n; j++) {
        if (x[j] !== 0.0 || y[j] !== 0.0) {
          const temp1 = alpha * y[j];
          const temp2 = alpha * x[j];
          for (let i = j; i < n; i++) {
            A[i + j * ldA] = A[i + j * ldA] + x[i] * temp1 + y[i] * temp2;
          }
        }
      }
    } else {
      let jx = kx;
      let jy = ky;
      for (let j = 0; j < n; j++) {
        if (x[jx] !== 0.0 || y[jy] !== 0.0) {
          const temp1 = alpha * y[jy];
          const temp2 = alpha * x[jx];
          let ix = jx;
          let iy = jy;
          for (let i = j; i < n; i++) {
            A[i + j * ldA] = A[i + j * ldA] + x[ix] * temp1 + y[iy] * temp2;
            ix += incx;
            iy += incy;
          }
        }
        jx += incx;
        jy += incy;
      }
    }
  }
}
