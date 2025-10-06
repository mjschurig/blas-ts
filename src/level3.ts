import { Transpose, Triangular, Diagonal, Side } from "./types";

export function dgemm(
  transA: Transpose,
  transB: Transpose,
  m: number,
  n: number,
  k: number,
  alpha: number,
  A: Float64Array,
  ldA: number,
  B: Float64Array,
  ldB: number,
  beta: number,
  C: Float64Array,
  ldC: number
): void {
  // ✅ IMPLEMENTED: Following ./packages/numeric/reference-implementation/BLAS/SRC/dgemm.f
  // DGEMM: performs C := alpha*op(A)*op(B) + beta*C
  // where op(X) = X or X^T

  // Set NOTA and NOTB as true if A and B respectively are not transposed
  // and set NROWA and NROWB as the number of rows of A and B respectively
  const notA = transA === Transpose.NoTranspose;
  const notB = transB === Transpose.NoTranspose;
  const nrowA = notA ? m : k;
  const nrowB = notB ? k : n;

  // Input validation
  if (
    transA !== Transpose.NoTranspose &&
    transA !== Transpose.Transpose &&
    transA !== Transpose.ConjugateTranspose
  ) {
    throw new Error("DGEMM: Invalid TRANSA parameter");
  }
  if (
    transB !== Transpose.NoTranspose &&
    transB !== Transpose.Transpose &&
    transB !== Transpose.ConjugateTranspose
  ) {
    throw new Error("DGEMM: Invalid TRANSB parameter");
  }
  if (m < 0) throw new Error("DGEMM: M must be >= 0");
  if (n < 0) throw new Error("DGEMM: N must be >= 0");
  if (k < 0) throw new Error("DGEMM: K must be >= 0");
  if (ldA < Math.max(1, nrowA))
    throw new Error("DGEMM: LDA must be >= max(1, nrowA)");
  if (ldB < Math.max(1, nrowB))
    throw new Error("DGEMM: LDB must be >= max(1, nrowB)");
  if (ldC < Math.max(1, m)) throw new Error("DGEMM: LDC must be >= max(1, M)");

  // Quick return if possible
  if (m === 0 || n === 0 || ((alpha === 0.0 || k === 0) && beta === 1.0)) {
    return;
  }

  // And if alpha equals zero
  if (alpha === 0.0) {
    if (beta === 0.0) {
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < m; i++) {
          C[i + j * ldC] = 0.0;
        }
      }
    } else {
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < m; i++) {
          C[i + j * ldC] = beta * C[i + j * ldC];
        }
      }
    }
    return;
  }

  // Start the operations
  if (notB) {
    if (notA) {
      // Form C := alpha*A*B + beta*C
      for (let j = 0; j < n; j++) {
        if (beta === 0.0) {
          for (let i = 0; i < m; i++) {
            C[i + j * ldC] = 0.0;
          }
        } else if (beta !== 1.0) {
          for (let i = 0; i < m; i++) {
            C[i + j * ldC] = beta * C[i + j * ldC];
          }
        }
        for (let l = 0; l < k; l++) {
          const temp = alpha * B[l + j * ldB];
          for (let i = 0; i < m; i++) {
            C[i + j * ldC] = C[i + j * ldC] + temp * A[i + l * ldA];
          }
        }
      }
    } else {
      // Form C := alpha*A^T*B + beta*C
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < m; i++) {
          let temp = 0.0;
          for (let l = 0; l < k; l++) {
            temp = temp + A[l + i * ldA] * B[l + j * ldB];
          }
          if (beta === 0.0) {
            C[i + j * ldC] = alpha * temp;
          } else {
            C[i + j * ldC] = alpha * temp + beta * C[i + j * ldC];
          }
        }
      }
    }
  } else {
    if (notA) {
      // Form C := alpha*A*B^T + beta*C
      for (let j = 0; j < n; j++) {
        if (beta === 0.0) {
          for (let i = 0; i < m; i++) {
            C[i + j * ldC] = 0.0;
          }
        } else if (beta !== 1.0) {
          for (let i = 0; i < m; i++) {
            C[i + j * ldC] = beta * C[i + j * ldC];
          }
        }
        for (let l = 0; l < k; l++) {
          const temp = alpha * B[j + l * ldB];
          for (let i = 0; i < m; i++) {
            C[i + j * ldC] = C[i + j * ldC] + temp * A[i + l * ldA];
          }
        }
      }
    } else {
      // Form C := alpha*A^T*B^T + beta*C
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < m; i++) {
          let temp = 0.0;
          for (let l = 0; l < k; l++) {
            temp = temp + A[l + i * ldA] * B[j + l * ldB];
          }
          if (beta === 0.0) {
            C[i + j * ldC] = alpha * temp;
          } else {
            C[i + j * ldC] = alpha * temp + beta * C[i + j * ldC];
          }
        }
      }
    }
  }
}

export function dsymm(
  side: Side,
  uplo: Triangular,
  m: number,
  n: number,
  alpha: number,
  A: Float64Array,
  ldA: number,
  B: Float64Array,
  ldB: number,
  beta: number,
  C: Float64Array,
  ldC: number
): void {
  // ✅ IMPLEMENTED: Following ./packages/numeric/reference-implementation/BLAS/SRC/dsymm.f
  // DSYMM: performs C := alpha*A*B + beta*C or C := alpha*B*A + beta*C
  // where A is a symmetric matrix

  // Set NROWA as the number of rows of A
  const nrowA = side === Side.Left ? m : n;
  const upper = uplo === Triangular.Upper;

  // Input validation
  if (side !== Side.Left && side !== Side.Right) {
    throw new Error("DSYMM: Invalid SIDE parameter");
  }
  if (uplo !== Triangular.Upper && uplo !== Triangular.Lower) {
    throw new Error("DSYMM: Invalid UPLO parameter");
  }
  if (m < 0) throw new Error("DSYMM: M must be >= 0");
  if (n < 0) throw new Error("DSYMM: N must be >= 0");
  if (ldA < Math.max(1, nrowA))
    throw new Error("DSYMM: LDA must be >= max(1, nrowA)");
  if (ldB < Math.max(1, m)) throw new Error("DSYMM: LDB must be >= max(1, M)");
  if (ldC < Math.max(1, m)) throw new Error("DSYMM: LDC must be >= max(1, M)");

  // Quick return if possible
  if (m === 0 || n === 0 || (alpha === 0.0 && beta === 1.0)) {
    return;
  }

  // And when alpha equals zero
  if (alpha === 0.0) {
    if (beta === 0.0) {
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < m; i++) {
          C[i + j * ldC] = 0.0;
        }
      }
    } else {
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < m; i++) {
          C[i + j * ldC] = beta * C[i + j * ldC];
        }
      }
    }
    return;
  }

  // Start the operations
  if (side === Side.Left) {
    // Form C := alpha*A*B + beta*C
    if (upper) {
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < m; i++) {
          const temp1 = alpha * B[i + j * ldB];
          let temp2 = 0.0;
          for (let k = 0; k < i; k++) {
            C[k + j * ldC] = C[k + j * ldC] + temp1 * A[k + i * ldA];
            temp2 = temp2 + B[k + j * ldB] * A[k + i * ldA];
          }
          if (beta === 0.0) {
            C[i + j * ldC] = temp1 * A[i + i * ldA] + alpha * temp2;
          } else {
            C[i + j * ldC] =
              beta * C[i + j * ldC] + temp1 * A[i + i * ldA] + alpha * temp2;
          }
        }
      }
    } else {
      for (let j = 0; j < n; j++) {
        for (let i = m - 1; i >= 0; i--) {
          const temp1 = alpha * B[i + j * ldB];
          let temp2 = 0.0;
          for (let k = i + 1; k < m; k++) {
            C[k + j * ldC] = C[k + j * ldC] + temp1 * A[k + i * ldA];
            temp2 = temp2 + B[k + j * ldB] * A[k + i * ldA];
          }
          if (beta === 0.0) {
            C[i + j * ldC] = temp1 * A[i + i * ldA] + alpha * temp2;
          } else {
            C[i + j * ldC] =
              beta * C[i + j * ldC] + temp1 * A[i + i * ldA] + alpha * temp2;
          }
        }
      }
    }
  } else {
    // Form C := alpha*B*A + beta*C
    for (let j = 0; j < n; j++) {
      const temp1 = alpha * A[j + j * ldA];
      if (beta === 0.0) {
        for (let i = 0; i < m; i++) {
          C[i + j * ldC] = temp1 * B[i + j * ldB];
        }
      } else {
        for (let i = 0; i < m; i++) {
          C[i + j * ldC] = beta * C[i + j * ldC] + temp1 * B[i + j * ldB];
        }
      }
      for (let k = 0; k < j; k++) {
        const temp1 = upper ? alpha * A[k + j * ldA] : alpha * A[j + k * ldA];
        for (let i = 0; i < m; i++) {
          C[i + j * ldC] = C[i + j * ldC] + temp1 * B[i + k * ldB];
        }
      }
      for (let k = j + 1; k < n; k++) {
        const temp1 = upper ? alpha * A[j + k * ldA] : alpha * A[k + j * ldA];
        for (let i = 0; i < m; i++) {
          C[i + j * ldC] = C[i + j * ldC] + temp1 * B[i + k * ldB];
        }
      }
    }
  }
}

export function dtrmm(
  side: Side,
  uplo: Triangular,
  transA: Transpose,
  diag: Diagonal,
  m: number,
  n: number,
  alpha: number,
  A: Float64Array,
  ldA: number,
  B: Float64Array,
  ldB: number
): void {
  // ✅ IMPLEMENTED: Following ./packages/numeric/reference-implementation/BLAS/SRC/dtrmm.f
  // DTRMM: performs B := alpha*op(A)*B or B := alpha*B*op(A)
  // where A is triangular

  // Test the input parameters
  const lside = side === Side.Left;
  const nrowA = lside ? m : n;
  const nounit = diag === Diagonal.NonUnit;
  const upper = uplo === Triangular.Upper;

  // Input validation
  if (side !== Side.Left && side !== Side.Right) {
    throw new Error("DTRMM: Invalid SIDE parameter");
  }
  if (uplo !== Triangular.Upper && uplo !== Triangular.Lower) {
    throw new Error("DTRMM: Invalid UPLO parameter");
  }
  if (
    transA !== Transpose.NoTranspose &&
    transA !== Transpose.Transpose &&
    transA !== Transpose.ConjugateTranspose
  ) {
    throw new Error("DTRMM: Invalid TRANSA parameter");
  }
  if (diag !== Diagonal.Unit && diag !== Diagonal.NonUnit) {
    throw new Error("DTRMM: Invalid DIAG parameter");
  }
  if (m < 0) throw new Error("DTRMM: M must be >= 0");
  if (n < 0) throw new Error("DTRMM: N must be >= 0");
  if (ldA < Math.max(1, nrowA))
    throw new Error("DTRMM: LDA must be >= max(1, nrowA)");
  if (ldB < Math.max(1, m)) throw new Error("DTRMM: LDB must be >= max(1, M)");

  // Quick return if possible
  if (m === 0 || n === 0) return;

  // And when alpha equals zero
  if (alpha === 0.0) {
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < m; i++) {
        B[i + j * ldB] = 0.0;
      }
    }
    return;
  }

  // Start the operations
  if (lside) {
    if (transA === Transpose.NoTranspose) {
      // Form B := alpha*A*B
      if (upper) {
        for (let j = 0; j < n; j++) {
          for (let k = 0; k < m; k++) {
            if (B[k + j * ldB] !== 0.0) {
              let temp = alpha * B[k + j * ldB];
              for (let i = 0; i < k; i++) {
                B[i + j * ldB] = B[i + j * ldB] + temp * A[i + k * ldA];
              }
              if (nounit) temp = temp * A[k + k * ldA];
              B[k + j * ldB] = temp;
            }
          }
        }
      } else {
        for (let j = 0; j < n; j++) {
          for (let k = m - 1; k >= 0; k--) {
            if (B[k + j * ldB] !== 0.0) {
              let temp = alpha * B[k + j * ldB];
              B[k + j * ldB] = temp;
              if (nounit) B[k + j * ldB] = B[k + j * ldB] * A[k + k * ldA];
              for (let i = k + 1; i < m; i++) {
                B[i + j * ldB] = B[i + j * ldB] + temp * A[i + k * ldA];
              }
            }
          }
        }
      }
    } else {
      // Form B := alpha*A^T*B
      if (upper) {
        for (let j = 0; j < n; j++) {
          for (let i = m - 1; i >= 0; i--) {
            let temp = B[i + j * ldB];
            if (nounit) temp = temp * A[i + i * ldA];
            for (let k = 0; k < i; k++) {
              temp = temp + A[k + i * ldA] * B[k + j * ldB];
            }
            B[i + j * ldB] = alpha * temp;
          }
        }
      } else {
        for (let j = 0; j < n; j++) {
          for (let i = 0; i < m; i++) {
            let temp = B[i + j * ldB];
            if (nounit) temp = temp * A[i + i * ldA];
            for (let k = i + 1; k < m; k++) {
              temp = temp + A[k + i * ldA] * B[k + j * ldB];
            }
            B[i + j * ldB] = alpha * temp;
          }
        }
      }
    }
  } else {
    if (transA === Transpose.NoTranspose) {
      // Form B := alpha*B*A
      if (upper) {
        for (let j = n - 1; j >= 0; j--) {
          let temp = alpha;
          if (nounit) temp = temp * A[j + j * ldA];
          for (let i = 0; i < m; i++) {
            B[i + j * ldB] = temp * B[i + j * ldB];
          }
          for (let k = 0; k < j; k++) {
            if (A[k + j * ldA] !== 0.0) {
              temp = alpha * A[k + j * ldA];
              for (let i = 0; i < m; i++) {
                B[i + j * ldB] = B[i + j * ldB] + temp * B[i + k * ldB];
              }
            }
          }
        }
      } else {
        for (let j = 0; j < n; j++) {
          let temp = alpha;
          if (nounit) temp = temp * A[j + j * ldA];
          for (let i = 0; i < m; i++) {
            B[i + j * ldB] = temp * B[i + j * ldB];
          }
          for (let k = j + 1; k < n; k++) {
            if (A[k + j * ldA] !== 0.0) {
              temp = alpha * A[k + j * ldA];
              for (let i = 0; i < m; i++) {
                B[i + j * ldB] = B[i + j * ldB] + temp * B[i + k * ldB];
              }
            }
          }
        }
      }
    } else {
      // Form B := alpha*B*A^T
      if (upper) {
        for (let k = 0; k < n; k++) {
          for (let j = 0; j < k; j++) {
            if (A[j + k * ldA] !== 0.0) {
              const temp = alpha * A[j + k * ldA];
              for (let i = 0; i < m; i++) {
                B[i + j * ldB] = B[i + j * ldB] + temp * B[i + k * ldB];
              }
            }
          }
          let temp = alpha;
          if (nounit) temp = temp * A[k + k * ldA];
          if (temp !== 1.0) {
            for (let i = 0; i < m; i++) {
              B[i + k * ldB] = temp * B[i + k * ldB];
            }
          }
        }
      } else {
        for (let k = n - 1; k >= 0; k--) {
          for (let j = k + 1; j < n; j++) {
            if (A[j + k * ldA] !== 0.0) {
              const temp = alpha * A[j + k * ldA];
              for (let i = 0; i < m; i++) {
                B[i + j * ldB] = B[i + j * ldB] + temp * B[i + k * ldB];
              }
            }
          }
          let temp = alpha;
          if (nounit) temp = temp * A[k + k * ldA];
          if (temp !== 1.0) {
            for (let i = 0; i < m; i++) {
              B[i + k * ldB] = temp * B[i + k * ldB];
            }
          }
        }
      }
    }
  }
}

export function dtrsm(
  side: Side,
  uplo: Triangular,
  transA: Transpose,
  diag: Diagonal,
  m: number,
  n: number,
  alpha: number,
  A: Float64Array,
  ldA: number,
  B: Float64Array,
  ldB: number
): void {
  // ✅ IMPLEMENTED: Following ./packages/numeric/reference-implementation/BLAS/SRC/dtrsm.f
  // DTRSM: solves op(A)*X = alpha*B or X*op(A) = alpha*B
  // where A is triangular and X is overwritten on B

  // Test the input parameters
  const lside = side === Side.Left;
  const nounit = diag === Diagonal.NonUnit;
  const upper = uplo === Triangular.Upper;

  // Quick return if possible
  if (m === 0 || n === 0) return;

  // And when alpha equals zero
  if (alpha === 0.0) {
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < m; i++) {
        B[i + j * ldB] = 0.0;
      }
    }
    return;
  }

  // Start the operations
  if (lside) {
    if (transA === Transpose.NoTranspose) {
      // Form B := alpha*inv(A)*B
      if (upper) {
        for (let j = 0; j < n; j++) {
          if (alpha !== 1.0) {
            for (let i = 0; i < m; i++) {
              B[i + j * ldB] = alpha * B[i + j * ldB];
            }
          }
          for (let k = m - 1; k >= 0; k--) {
            if (B[k + j * ldB] !== 0.0) {
              if (nounit) B[k + j * ldB] = B[k + j * ldB] / A[k + k * ldA];
              for (let i = 0; i < k; i++) {
                B[i + j * ldB] =
                  B[i + j * ldB] - B[k + j * ldB] * A[i + k * ldA];
              }
            }
          }
        }
      } else {
        for (let j = 0; j < n; j++) {
          if (alpha !== 1.0) {
            for (let i = 0; i < m; i++) {
              B[i + j * ldB] = alpha * B[i + j * ldB];
            }
          }
          for (let k = 0; k < m; k++) {
            if (B[k + j * ldB] !== 0.0) {
              if (nounit) B[k + j * ldB] = B[k + j * ldB] / A[k + k * ldA];
              for (let i = k + 1; i < m; i++) {
                B[i + j * ldB] =
                  B[i + j * ldB] - B[k + j * ldB] * A[i + k * ldA];
              }
            }
          }
        }
      }
    } else {
      // Form B := alpha*inv(A^T)*B
      if (upper) {
        for (let j = 0; j < n; j++) {
          for (let i = 0; i < m; i++) {
            let temp = alpha * B[i + j * ldB];
            for (let k = 0; k < i; k++) {
              temp = temp - A[k + i * ldA] * B[k + j * ldB];
            }
            if (nounit) temp = temp / A[i + i * ldA];
            B[i + j * ldB] = temp;
          }
        }
      } else {
        for (let j = 0; j < n; j++) {
          for (let i = 0; i < m; i++) {
            let temp = alpha * B[i + j * ldB];
            for (let k = 0; k < i; k++) {
              temp = temp - A[i + k * ldA] * B[k + j * ldB];
            }
            if (nounit) temp = temp / A[i + i * ldA];
            B[i + j * ldB] = temp;
          }
        }
      }
    }
  } else {
    if (transA === Transpose.NoTranspose) {
      // Form B := alpha*B*inv(A)
      if (upper) {
        for (let j = 0; j < n; j++) {
          if (alpha !== 1.0) {
            for (let i = 0; i < m; i++) {
              B[i + j * ldB] = alpha * B[i + j * ldB];
            }
          }
          for (let k = 0; k < j; k++) {
            if (A[k + j * ldA] !== 0.0) {
              for (let i = 0; i < m; i++) {
                B[i + j * ldB] =
                  B[i + j * ldB] - A[k + j * ldA] * B[i + k * ldB];
              }
            }
          }
          if (nounit) {
            const temp = 1.0 / A[j + j * ldA];
            for (let i = 0; i < m; i++) {
              B[i + j * ldB] = temp * B[i + j * ldB];
            }
          }
        }
      } else {
        for (let j = n - 1; j >= 0; j--) {
          if (alpha !== 1.0) {
            for (let i = 0; i < m; i++) {
              B[i + j * ldB] = alpha * B[i + j * ldB];
            }
          }
          for (let k = j + 1; k < n; k++) {
            if (A[k + j * ldA] !== 0.0) {
              for (let i = 0; i < m; i++) {
                B[i + j * ldB] =
                  B[i + j * ldB] - A[k + j * ldA] * B[i + k * ldB];
              }
            }
          }
          if (nounit) {
            const temp = 1.0 / A[j + j * ldA];
            for (let i = 0; i < m; i++) {
              B[i + j * ldB] = temp * B[i + j * ldB];
            }
          }
        }
      }
    } else {
      // Form B := alpha*B*inv(A^T)
      if (upper) {
        for (let k = n - 1; k >= 0; k--) {
          if (nounit) {
            const temp = 1.0 / A[k + k * ldA];
            for (let i = 0; i < m; i++) {
              B[i + k * ldB] = temp * B[i + k * ldB];
            }
          }
          for (let j = 0; j < k; j++) {
            if (A[j + k * ldA] !== 0.0) {
              const temp = A[j + k * ldA];
              for (let i = 0; i < m; i++) {
                B[i + j * ldB] = B[i + j * ldB] - temp * B[i + k * ldB];
              }
            }
          }
          if (alpha !== 1.0) {
            for (let i = 0; i < m; i++) {
              B[i + k * ldB] = alpha * B[i + k * ldB];
            }
          }
        }
      } else {
        for (let k = 0; k < n; k++) {
          if (nounit) {
            const temp = 1.0 / A[k + k * ldA];
            for (let i = 0; i < m; i++) {
              B[i + k * ldB] = temp * B[i + k * ldB];
            }
          }
          for (let j = k + 1; j < n; j++) {
            if (A[j + k * ldA] !== 0.0) {
              const temp = A[j + k * ldA];
              for (let i = 0; i < m; i++) {
                B[i + j * ldB] = B[i + j * ldB] - temp * B[i + k * ldB];
              }
            }
          }
          if (alpha !== 1.0) {
            for (let i = 0; i < m; i++) {
              B[i + k * ldB] = alpha * B[i + k * ldB];
            }
          }
        }
      }
    }
  }
}

export function dsyrk(
  uplo: Triangular,
  trans: Transpose,
  n: number,
  k: number,
  alpha: number,
  A: Float64Array,
  ldA: number,
  beta: number,
  C: Float64Array,
  ldC: number
): void {
  // DSYRK: performs one of the symmetric rank k operations
  // C := alpha*A*A^T + beta*C  or  C := alpha*A^T*A + beta*C

  // Determine the number of rows of A
  const nrowA = trans === Transpose.NoTranspose ? n : k;
  const upper = uplo === Triangular.Upper;

  // Input validation
  if (uplo !== Triangular.Upper && uplo !== Triangular.Lower) {
    throw new Error("DSYRK: Invalid UPLO parameter");
  }
  if (
    trans !== Transpose.NoTranspose &&
    trans !== Transpose.Transpose &&
    trans !== Transpose.ConjugateTranspose
  ) {
    throw new Error("DSYRK: Invalid TRANS parameter");
  }
  if (n < 0) throw new Error("DSYRK: N must be >= 0");
  if (k < 0) throw new Error("DSYRK: K must be >= 0");
  if (ldA < Math.max(1, nrowA))
    throw new Error("DSYRK: LDA must be >= max(1, nrowA)");
  if (ldC < Math.max(1, n)) throw new Error("DSYRK: LDC must be >= max(1, N)");

  // Quick return if possible
  if (n === 0 || ((alpha === 0.0 || k === 0) && beta === 1.0)) {
    return;
  }

  // And when alpha equals zero
  if (alpha === 0.0) {
    if (upper) {
      if (beta === 0.0) {
        for (let j = 0; j < n; j++) {
          for (let i = 0; i <= j; i++) {
            C[i + j * ldC] = 0.0;
          }
        }
      } else {
        for (let j = 0; j < n; j++) {
          for (let i = 0; i <= j; i++) {
            C[i + j * ldC] = beta * C[i + j * ldC];
          }
        }
      }
    } else {
      if (beta === 0.0) {
        for (let j = 0; j < n; j++) {
          for (let i = j; i < n; i++) {
            C[i + j * ldC] = 0.0;
          }
        }
      } else {
        for (let j = 0; j < n; j++) {
          for (let i = j; i < n; i++) {
            C[i + j * ldC] = beta * C[i + j * ldC];
          }
        }
      }
    }
    return;
  }

  // Start the operations
  if (trans === Transpose.NoTranspose) {
    // Form C := alpha*A*A^T + beta*C
    if (upper) {
      for (let j = 0; j < n; j++) {
        if (beta === 0.0) {
          for (let i = 0; i <= j; i++) {
            C[i + j * ldC] = 0.0;
          }
        } else if (beta !== 1.0) {
          for (let i = 0; i <= j; i++) {
            C[i + j * ldC] = beta * C[i + j * ldC];
          }
        }
        for (let l = 0; l < k; l++) {
          if (A[j + l * ldA] !== 0.0) {
            const temp = alpha * A[j + l * ldA];
            for (let i = 0; i <= j; i++) {
              C[i + j * ldC] = C[i + j * ldC] + temp * A[i + l * ldA];
            }
          }
        }
      }
    } else {
      for (let j = 0; j < n; j++) {
        if (beta === 0.0) {
          for (let i = j; i < n; i++) {
            C[i + j * ldC] = 0.0;
          }
        } else if (beta !== 1.0) {
          for (let i = j; i < n; i++) {
            C[i + j * ldC] = beta * C[i + j * ldC];
          }
        }
        for (let l = 0; l < k; l++) {
          if (A[j + l * ldA] !== 0.0) {
            const temp = alpha * A[j + l * ldA];
            for (let i = j; i < n; i++) {
              C[i + j * ldC] = C[i + j * ldC] + temp * A[i + l * ldA];
            }
          }
        }
      }
    }
  } else {
    // Form C := alpha*A^T*A + beta*C
    if (upper) {
      for (let j = 0; j < n; j++) {
        for (let i = 0; i <= j; i++) {
          let temp = 0.0;
          for (let l = 0; l < k; l++) {
            temp = temp + A[l + i * ldA] * A[l + j * ldA];
          }
          if (beta === 0.0) {
            C[i + j * ldC] = alpha * temp;
          } else {
            C[i + j * ldC] = alpha * temp + beta * C[i + j * ldC];
          }
        }
      }
    } else {
      for (let j = 0; j < n; j++) {
        for (let i = j; i < n; i++) {
          let temp = 0.0;
          for (let l = 0; l < k; l++) {
            temp = temp + A[l + i * ldA] * A[l + j * ldA];
          }
          if (beta === 0.0) {
            C[i + j * ldC] = alpha * temp;
          } else {
            C[i + j * ldC] = alpha * temp + beta * C[i + j * ldC];
          }
        }
      }
    }
  }
}

export function dsyr2k(
  uplo: Triangular,
  trans: Transpose,
  n: number,
  k: number,
  alpha: number,
  A: Float64Array,
  ldA: number,
  B: Float64Array,
  ldB: number,
  beta: number,
  C: Float64Array,
  ldC: number
): void {
  // DSYR2K: performs one of the symmetric rank 2k operations
  // C := alpha*A*B^T + alpha*B*A^T + beta*C  or  C := alpha*A^T*B + alpha*B^T*A + beta*C

  // Determine the number of rows of A and B
  const nrowA = trans === Transpose.NoTranspose ? n : k;
  const upper = uplo === Triangular.Upper;

  // Input validation
  if (uplo !== Triangular.Upper && uplo !== Triangular.Lower) {
    throw new Error("DSYR2K: Invalid UPLO parameter");
  }
  if (
    trans !== Transpose.NoTranspose &&
    trans !== Transpose.Transpose &&
    trans !== Transpose.ConjugateTranspose
  ) {
    throw new Error("DSYR2K: Invalid TRANS parameter");
  }
  if (n < 0) throw new Error("DSYR2K: N must be >= 0");
  if (k < 0) throw new Error("DSYR2K: K must be >= 0");
  if (ldA < Math.max(1, nrowA))
    throw new Error("DSYR2K: LDA must be >= max(1, nrowA)");
  if (ldB < Math.max(1, nrowA))
    throw new Error("DSYR2K: LDB must be >= max(1, nrowA)");
  if (ldC < Math.max(1, n)) throw new Error("DSYR2K: LDC must be >= max(1, N)");

  // Quick return if possible
  if (n === 0 || ((alpha === 0.0 || k === 0) && beta === 1.0)) {
    return;
  }

  const index = (i: number, j: number) => i + j * ldC;

  // And when alpha equals zero
  if (alpha === 0.0) {
    if (upper) {
      if (beta === 0.0) {
        for (let j = 0; j < n; j++) {
          for (let i = 0; i <= j; i++) {
            C[index(i, j)] = 0.0;
          }
        }
      } else {
        for (let j = 0; j < n; j++) {
          for (let i = 0; i <= j; i++) {
            C[i + j * ldC] = beta * C[i + j * ldC];
          }
        }
      }
    } else {
      if (beta === 0.0) {
        for (let j = 0; j < n; j++) {
          for (let i = j; i < n; i++) {
            C[i + j * ldC] = 0.0;
          }
        }
      } else {
        for (let j = 0; j < n; j++) {
          for (let i = j; i < n; i++) {
            C[i + j * ldC] = beta * C[i + j * ldC];
          }
        }
      }
    }
    return;
  }

  // Start the operations
  if (trans === Transpose.NoTranspose) {
    // Form C := alpha*A*B^T + alpha*B*A^T + beta*C
    if (upper) {
      for (let j = 0; j < n; j++) {
        if (beta === 0.0) {
          for (let i = 0; i <= j; i++) {
            C[i + j * ldC] = 0.0;
          }
        } else if (beta !== 1.0) {
          for (let i = 0; i <= j; i++) {
            C[i + j * ldC] = beta * C[i + j * ldC];
          }
        }
        for (let l = 0; l < k; l++) {
          if (A[j + l * ldA] !== 0.0 || B[j + l * ldB] !== 0.0) {
            const temp1 = alpha * B[j + l * ldB];
            const temp2 = alpha * A[j + l * ldA];
            for (let i = 0; i <= j; i++) {
              C[i + j * ldC] =
                C[i + j * ldC] +
                A[i + l * ldA] * temp1 +
                B[i + l * ldB] * temp2;
            }
          }
        }
      }
    } else {
      for (let j = 0; j < n; j++) {
        if (beta === 0.0) {
          for (let i = j; i < n; i++) {
            C[i + j * ldC] = 0.0;
          }
        } else if (beta !== 1.0) {
          for (let i = j; i < n; i++) {
            C[i + j * ldC] = beta * C[i + j * ldC];
          }
        }
        for (let l = 0; l < k; l++) {
          if (A[j + l * ldA] !== 0.0 || B[j + l * ldB] !== 0.0) {
            const temp1 = alpha * B[j + l * ldB];
            const temp2 = alpha * A[j + l * ldA];
            for (let i = j; i < n; i++) {
              C[i + j * ldC] =
                C[i + j * ldC] +
                A[i + l * ldA] * temp1 +
                B[i + l * ldB] * temp2;
            }
          }
        }
      }
    }
  } else {
    // Form C := alpha*A^T*B + alpha*B^T*A + beta*C
    if (upper) {
      for (let j = 0; j < n; j++) {
        for (let i = 0; i <= j; i++) {
          let temp1 = 0.0;
          let temp2 = 0.0;
          for (let l = 0; l < k; l++) {
            temp1 = temp1 + A[l + i * ldA] * B[l + j * ldB];
            temp2 = temp2 + B[l + i * ldB] * A[l + j * ldA];
          }
          if (beta === 0.0) {
            C[i + j * ldC] = alpha * temp1 + alpha * temp2;
          } else {
            C[i + j * ldC] =
              beta * C[i + j * ldC] + alpha * temp1 + alpha * temp2;
          }
        }
      }
    } else {
      for (let j = 0; j < n; j++) {
        for (let i = j; i < n; i++) {
          let temp1 = 0.0;
          let temp2 = 0.0;
          for (let l = 0; l < k; l++) {
            temp1 = temp1 + A[l + i * ldA] * B[l + j * ldB];
            temp2 = temp2 + B[l + i * ldB] * A[l + j * ldA];
          }
          if (beta === 0.0) {
            C[i + j * ldC] = alpha * temp1 + alpha * temp2;
          } else {
            C[i + j * ldC] =
              beta * C[i + j * ldC] + alpha * temp1 + alpha * temp2;
          }
        }
      }
    }
  }
}
