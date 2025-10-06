/**
 * DAXPY: constant times a vector plus a vector (y = alpha*x + y)
 *
 * @param n - number of elements to process
 * @param alpha - scalar multiplier
 * @param x - input vector x
 * @param y - input/output vector y (modified in place)
 */
export function daxpy(
  n: number,
  alpha: number,
  x: Float64Array,
  incx: number,
  y: Float64Array,
  incy: number
): void {
  // Early returns
  if (n <= 0) return;
  if (alpha === 0.0) return;

  if (incx === 1 && incy === 1) {
    // Optimized code for both increments equal to 1
    // Clean-up loop for unrolling by 4
    const m = n % 4;

    if (m !== 0) {
      for (let i = 0; i < m; i++) {
        y[i] = y[i] + alpha * x[i];
      }
    }

    if (n < 4) return;

    // Unrolled loop by 4 for better performance
    for (let i = m; i < n; i += 4) {
      y[i] = y[i] + alpha * x[i];
      y[i + 1] = y[i + 1] + alpha * x[i + 1];
      y[i + 2] = y[i + 2] + alpha * x[i + 2];
      y[i + 3] = y[i + 3] + alpha * x[i + 3];
    }
  } else {
    // Code for unequal increments or equal increments not equal to 1
    let ix = incx < 0 ? (-n + 1) * incx : 0;
    let iy = incy < 0 ? (-n + 1) * incy : 0;

    for (let i = 0; i < n; i++) {
      y[iy] = y[iy] + alpha * x[ix];
      ix += incx;
      iy += incy;
    }
  }
}

/**
 * DSCAL: scale a vector by a constant (x = alpha * x)
 *
 * @param alpha - scalar multiplier
 * @param x - input/output vector x (modified in place)
 */
export function dscal(
  n: number,
  alpha: number,
  x: Float64Array,
  incx: number
): void {
  // Early returns
  if (n <= 0 || incx <= 0 || alpha === 1.0) return;

  if (incx === 1) {
    // Optimized code for increment equal to 1
    // Clean-up loop for unrolling by 5
    const m = n % 5;

    if (m !== 0) {
      for (let i = 0; i < m; i++) {
        x[i] = alpha * x[i];
      }
      if (n < 5) return;
    }

    // Unrolled loop by 5 for better performance
    for (let i = m; i < n; i += 5) {
      x[i] = alpha * x[i];
      x[i + 1] = alpha * x[i + 1];
      x[i + 2] = alpha * x[i + 2];
      x[i + 3] = alpha * x[i + 3];
      x[i + 4] = alpha * x[i + 4];
    }
  } else {
    // Code for increment not equal to 1
    const nincx = n * incx;
    for (let i = 0; i < nincx; i += incx) {
      x[i] = alpha * x[i];
    }
  }
}

/**
 * DCOPY: copy a vector (y = x)
 *
 * @param x - input vector x
 * @param y - input/output vector y (modified in place)
 */
export function dcopy(
  n: number,
  x: Float64Array,
  incx: number,
  y: Float64Array,
  incy: number
): void {
  // Early return
  if (n <= 0) return;

  if (incx === 1 && incy === 1) {
    // Optimized code for both increments equal to 1
    // Clean-up loop for unrolling by 7
    const m = n % 7;

    if (m !== 0) {
      for (let i = 0; i < m; i++) {
        y[i] = x[i];
      }
      if (n < 7) return;
    }

    // Unrolled loop by 7 for better performance
    for (let i = m; i < n; i += 7) {
      y[i] = x[i];
      y[i + 1] = x[i + 1];
      y[i + 2] = x[i + 2];
      y[i + 3] = x[i + 3];
      y[i + 4] = x[i + 4];
      y[i + 5] = x[i + 5];
      y[i + 6] = x[i + 6];
    }
  } else {
    // Code for unequal increments or equal increments not equal to 1
    let ix = incx < 0 ? (-n + 1) * incx : 0;
    let iy = incy < 0 ? (-n + 1) * incy : 0;

    for (let i = 0; i < n; i++) {
      y[iy] = x[ix];
      ix += incx;
      iy += incy;
    }
  }
}

/**
 * DSWAP: swap two vectors (x <-> y)
 *
 * @param x - input/output vector x (modified in place)
 * @param y - input/output vector y (modified in place)
 */
export function dswap(
  n: number,
  x: Float64Array,
  incx: number,
  y: Float64Array,
  incy: number
): void {
  // Early return
  if (n <= 0) return;

  if (incx === 1 && incy === 1) {
    // Optimized code for both increments equal to 1
    // Clean-up loop for unrolling by 3
    const m = n % 3;

    if (m !== 0) {
      for (let i = 0; i < m; i++) {
        const temp = x[i];
        x[i] = y[i];
        y[i] = temp;
      }
      if (n < 3) return;
    }

    // Unrolled loop by 3 for better performance
    for (let i = m; i < n; i += 3) {
      // Swap element i
      let temp = x[i];
      x[i] = y[i];
      y[i] = temp;

      // Swap element i+1
      temp = x[i + 1];
      x[i + 1] = y[i + 1];
      y[i + 1] = temp;

      // Swap element i+2
      temp = x[i + 2];
      x[i + 2] = y[i + 2];
      y[i + 2] = temp;
    }
  } else {
    // Code for unequal increments or equal increments not equal to 1
    let ix = incx < 0 ? (-n + 1) * incx : 0;
    let iy = incy < 0 ? (-n + 1) * incy : 0;

    for (let i = 0; i < n; i++) {
      const temp = x[ix];
      x[ix] = y[iy];
      y[iy] = temp;
      ix += incx;
      iy += incy;
    }
  }
}

/**
 * DDOT: dot product of two vectors (x^T * y)
 *
 * @param x - input vector x
 * @param y - input vector y
 * @returns the dot product of x and y
 */
export function ddot(
  n: number,
  x: Float64Array,
  incx: number,
  y: Float64Array,
  incy: number
): number {
  let dtemp = 0.0;

  // Early return
  if (n <= 0) return 0.0;

  if (incx === 1 && incy === 1) {
    // Optimized code for both increments equal to 1
    // Clean-up loop for unrolling by 5
    const m = n % 5;

    if (m !== 0) {
      for (let i = 0; i < m; i++) {
        dtemp = dtemp + x[i] * y[i];
      }
      if (n < 5) {
        return dtemp;
      }
    }

    // Unrolled loop by 5 for better performance
    for (let i = m; i < n; i += 5) {
      dtemp =
        dtemp +
        x[i] * y[i] +
        x[i + 1] * y[i + 1] +
        x[i + 2] * y[i + 2] +
        x[i + 3] * y[i + 3] +
        x[i + 4] * y[i + 4];
    }
  } else {
    // Code for unequal increments or equal increments not equal to 1
    let ix = incx < 0 ? (-n + 1) * incx : 0;
    let iy = incy < 0 ? (-n + 1) * incy : 0;

    for (let i = 0; i < n; i++) {
      dtemp = dtemp + x[ix] * y[iy];
      ix += incx;
      iy += incy;
    }
  }

  return dtemp;
}

/**
 * DNRM2: Euclidean norm of a vector (sqrt(x^T * x))
 *
 * @param x - input vector x
 * @returns the Euclidean norm of x
 */
export function dnrm2(n: number, x: Float64Array, incx: number): number {
  // Early return
  if (n <= 0) return 0.0;

  let scale = 0.0;
  let ssq = 1.0;

  // Use simple algorithm with scaling to avoid overflow/underflow
  if (incx === 1) {
    // Optimized path for contiguous memory
    for (let i = 0; i < n; i++) {
      const absxi = Math.abs(x[i]);
      if (absxi !== 0.0) {
        if (scale < absxi) {
          const temp = scale / absxi;
          ssq = 1.0 + ssq * temp * temp;
          scale = absxi;
        } else {
          const temp = absxi / scale;
          ssq += temp * temp;
        }
      }
    }
  } else {
    // General increment case
    let ix = incx < 0 ? (-n + 1) * incx : 0;
    for (let i = 0; i < n; i++) {
      const absxi = Math.abs(x[ix]);
      if (absxi !== 0.0) {
        if (scale < absxi) {
          const temp = scale / absxi;
          ssq = 1.0 + ssq * temp * temp;
          scale = absxi;
        } else {
          const temp = absxi / scale;
          ssq += temp * temp;
        }
      }
      ix += incx;
    }
  }

  return scale * Math.sqrt(ssq);
}

/**
 * DASUM: sum of absolute values of a vector (|x_1| + |x_2| + ... + |x_n|)
 *
 * @param x - input vector x
 * @returns the sum of absolute values of x
 */
export function dasum(n: number, x: Float64Array, incx: number): number {
  let dtemp = 0.0;

  // Early returns
  if (n <= 0 || incx <= 0) return 0.0;

  if (incx === 1) {
    // Optimized code for increment equal to 1
    // Clean-up loop for unrolling by 6
    const m = n % 6;

    if (m !== 0) {
      for (let i = 0; i < m; i++) {
        dtemp += Math.abs(x[i]);
      }
      if (n < 6) {
        return dtemp;
      }
    }

    // Unrolled loop by 6 for better performance
    for (let i = m; i < n; i += 6) {
      dtemp +=
        Math.abs(x[i]) +
        Math.abs(x[i + 1]) +
        Math.abs(x[i + 2]) +
        Math.abs(x[i + 3]) +
        Math.abs(x[i + 4]) +
        Math.abs(x[i + 5]);
    }
  } else {
    // Code for increment not equal to 1
    const nincx = n * incx;
    for (let i = 0; i < nincx; i += incx) {
      dtemp += Math.abs(x[i]);
    }
  }

  return dtemp;
}

/**
 * IDAMAX: index of the element with the maximum absolute value
 *
 * @param x - input vector x
 * @returns the index of the element with the maximum absolute value
 */
export function idamax(n: number, x: Float64Array, incx: number): number {
  // Early returns
  if (n < 1 || incx <= 0) return 0;
  if (n === 1) return 0;

  let idamax = 0;

  if (incx === 1) {
    // Optimized code for increment equal to 1
    let dmax = Math.abs(x[0]);
    for (let i = 1; i < n; i++) {
      const absval = Math.abs(x[i]);
      if (absval > dmax) {
        idamax = i;
        dmax = absval;
      }
    }
  } else {
    // Code for increment not equal to 1
    let ix = 0;
    let dmax = Math.abs(x[0]);
    ix += incx;

    for (let i = 1; i < n; i++) {
      const absval = Math.abs(x[ix]);
      if (absval > dmax) {
        idamax = i;
        dmax = absval;
      }
      ix += incx;
    }
  }

  return idamax;
}

/**
 * DROTG: generate a Givens plane rotation (a, b) -> (c, s)
 *
 * Constructs the Givens rotation matrix G such that:
 * [ c  s] [a] = [r]
 * [-s  c] [b]   [0]
 *
 * @param a - first input scalar
 * @param b - second input scalar
 * @returns object with c (cosine), s (sine), r (resulting norm), and z (reconstruction info)
 */
export function drotg(
  a: number,
  b: number
): { c: number; s: number; r: number; z: number } {
  const anorm = Math.abs(a);
  const bnorm = Math.abs(b);

  let c: number, s: number, r: number, z: number;

  if (bnorm === 0.0) {
    c = 1.0;
    s = 0.0;
    r = a;
    z = 0.0;
  } else if (anorm === 0.0) {
    c = 0.0;
    s = 1.0;
    r = b;
    z = 1.0;
  } else {
    // Both a and b are non-zero
    const scale = Math.max(anorm, bnorm);
    const sigma = anorm > bnorm ? Math.sign(a) : Math.sign(b);

    // Scale to avoid overflow/underflow
    const as = a / scale;
    const bs = b / scale;

    r = sigma * scale * Math.sqrt(as * as + bs * bs);
    c = a / r;
    s = b / r;

    if (anorm > bnorm) {
      z = s;
    } else if (c !== 0.0) {
      z = 1.0 / c;
    } else {
      z = 1.0;
    }
  }

  return { c, s, r, z };
}

/**
 * DROT: apply a Givens plane rotation (x, y) -> (x', y')
 *
 * @param x - input/output vector x (modified in place)
 * @param y - input/output vector y (modified in place)
 * @param c - cosine of the rotation
 * @param s - sine of the rotation
 */
export function drot(
  n: number,
  x: Float64Array,
  incx: number,
  y: Float64Array,
  incy: number,
  c: number,
  s: number
): void {
  // Early return
  if (n <= 0) return;

  if (incx === 1 && incy === 1) {
    // Optimized code for both increments equal to 1
    for (let i = 0; i < n; i++) {
      const temp = c * x[i] + s * y[i];
      y[i] = c * y[i] - s * x[i];
      x[i] = temp;
    }
  } else {
    // Code for unequal increments or equal increments not equal to 1
    let ix = incx < 0 ? (-n + 1) * incx : 0;
    let iy = incy < 0 ? (-n + 1) * incy : 0;

    for (let i = 0; i < n; i++) {
      const temp = c * x[ix] + s * y[iy];
      y[iy] = c * y[iy] - s * x[ix];
      x[ix] = temp;
      ix += incx;
      iy += incy;
    }
  }
}
