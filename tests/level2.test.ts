import { dgemv, dger, dsymv, dsyr, dtrmv, dtrsv } from "../src/level2";
import { Diagonal, Transpose, Triangular } from "../src/types";

describe("BLAS Level 2 - Matrix-Vector Operations", () => {
  describe("DGEMV - General Matrix-Vector Multiply", () => {
    it("should compute y = alpha*A*x + beta*y correctly (no transpose)", () => {
      const m = 3,
        n = 2;
      const alpha = 2.0,
        beta = 0.5;
      const A = new Float64Array([1, 3, 5, 2, 4, 6]); // 3x2 matrix in column-major: [new Float64Array([1,2]),new Float64Array([3,4]),new Float64Array([5,6])]
      const x = new Float64Array([1, 2]); // length n = 2
      const y = new Float64Array([10, 20, 30]); // length m = 3
      const expected = new Float64Array([
        0.5 * 10 + 2.0 * (1 * 1 + 2 * 2), // 5 + 2*(5) = 15
        0.5 * 20 + 2.0 * (3 * 1 + 4 * 2), // 10 + 2*(11) = 32
        0.5 * 30 + 2.0 * (5 * 1 + 6 * 2), // 15 + 2*(17) = 49
      ]);

      dgemv(Transpose.NoTranspose, m, n, alpha, A, m, x, 1, beta, y, 1);

      expect(y).toEqual(expected);
    });

    it("should compute y = alpha*A^T*x + beta*y correctly (transpose)", () => {
      const m = 3,
        n = 2;
      const alpha = 1.0,
        beta = 0.0;
      const A = new Float64Array([1, 3, 5, 2, 4, 6]); // 3x2 matrix in column-major: [new Float64Array([1,2]),new Float64Array([3,4]),new Float64Array([5,6])]
      const x = new Float64Array([1, 2, 3]); // length m = 3
      const y = new Float64Array([0, 0]); // length n = 2
      // A^T*x = [1 3 5; 2 4 6] * [1; 2; 3] = [1*1+3*2+5*3; 2*1+4*2+6*3] = [22; 28]
      const expected = new Float64Array([22, 28]);

      dgemv(Transpose.Transpose, m, n, alpha, A, m, x, 1, beta, y, 1);

      expect(y).toEqual(expected);
    });

    it("should handle alpha = 0 (early return after beta*y)", () => {
      const m = 2,
        n = 2;
      const alpha = 0.0,
        beta = 2.0;
      const A = new Float64Array([1, 3, 2, 4]); // 2x2 matrix in column-major: [new Float64Array([1,2]),new Float64Array([3,4])]
      const x = new Float64Array([1, 1]);
      const y = new Float64Array([5, 10]);
      const expected = new Float64Array([10, 20]); // Just beta*y

      dgemv(Transpose.NoTranspose, m, n, alpha, A, m, x, 1, beta, y, 1);

      expect(y).toEqual(expected);
    });

    it("should handle beta = 0 (zero out y first)", () => {
      const m = 2,
        n = 2;
      const alpha = 1.0,
        beta = 0.0;
      const A = new Float64Array([1, 0, 0, 1]); // 2x2 identity matrix in column-major: [new Float64Array([1,0]),new Float64Array([0,1])]
      const x = new Float64Array([5, 7]);
      const y = new Float64Array([100, 200]); // Should be ignored due to beta = 0
      const expected = new Float64Array([5, 7]); // Just A*x

      dgemv(Transpose.NoTranspose, m, n, alpha, A, m, x, 1, beta, y, 1);

      expect(y).toEqual(expected);
    });

    it("should handle different increments", () => {
      const m = 2,
        n = 2;
      const alpha = 1.0,
        beta = 0.0;
      const A = new Float64Array([1, 3, 2, 4]); // 2x2 matrix in column-major: [new Float64Array([1,2]),new Float64Array([3,4])]
      const x = new Float64Array([1, 99, 2, 99]); // effective x = new Float64Array([1, 2]) with incx = 2
      const y = new Float64Array([0, 88, 0, 88]); // effective y = new Float64Array([0, 0]) with incy = 2
      // A*x = [1*1+2*2; 3*1+4*2] = [5; 11]
      const expected = new Float64Array([5, 88, 11, 88]);

      dgemv(Transpose.NoTranspose, m, n, alpha, A, m, x, 2, beta, y, 2);

      expect(y).toEqual(expected);
    });
  });

  describe("DSYMV - Symmetric Matrix-Vector Multiply", () => {
    it("should compute y = alpha*A*x + beta*y correctly (upper triangle)", () => {
      const n = 3;
      const alpha = 1.0,
        beta = 0.0;
      // Symmetric matrix stored in upper triangle (column-major):
      // [new Float64Array([1, 2, 3]),
      //  new Float64Array([2, 4, 5]),
      //  new Float64Array([3, 5, 6])]
      const A = new Float64Array([1, 2, 3, 2, 4, 5, 3, 5, 6]); // Only upper triangle is referenced
      const x = new Float64Array([1, 1, 1]);
      const y = new Float64Array([0, 0, 0]);
      // Full symmetric A*x = [1+2+3; 2+4+5; 3+5+6] = [6; 11; 14]
      const expected = new Float64Array([6, 11, 14]);

      dsymv(Triangular.Upper, n, alpha, A, n, x, 1, beta, y, 1);

      expect(y).toEqual(expected);
    });

    it("should compute y = alpha*A*x + beta*y correctly (lower triangle)", () => {
      const n = 3;
      const alpha = 1.0,
        beta = 0.0;
      // Symmetric matrix stored in lower triangle (column-major):
      // [new Float64Array([1, 2, 3]),
      //  new Float64Array([2, 4, 5]),
      //  new Float64Array([3, 5, 6])]
      const A = new Float64Array([1, 2, 3, 2, 4, 5, 3, 5, 6]); // Only lower triangle is referenced
      const x = new Float64Array([1, 1, 1]);
      const y = new Float64Array([0, 0, 0]);
      // Full symmetric A*x = [1+2+3; 2+4+5; 3+5+6] = [6; 11; 14]
      const expected = new Float64Array([6, 11, 14]);

      dsymv(Triangular.Lower, n, alpha, A, n, x, 1, beta, y, 1);

      expect(y).toEqual(expected);
    });

    it("should handle alpha = 0 and beta != 1", () => {
      const n = 2;
      const alpha = 0.0,
        beta = 3.0;
      const A = new Float64Array([1, 2, 2, 4]); // 2x2 symmetric matrix in column-major
      const x = new Float64Array([1, 1]);
      const y = new Float64Array([5, 7]);
      const expected = new Float64Array([15, 21]); // Just beta*y

      dsymv(Triangular.Upper, n, alpha, A, n, x, 1, beta, y, 1);

      expect(y).toEqual(expected);
    });

    it("should handle different increments", () => {
      const n = 2;
      const alpha = 1.0,
        beta = 0.0;
      const A = new Float64Array([1, 2, 2, 3]); // 2x2 symmetric matrix in column-major: [new Float64Array([1,2]),new Float64Array([2,3])]
      const x = new Float64Array([1, 99, 2, 99]); // effective x = new Float64Array([1, 2]) with incx = 2
      const y = new Float64Array([0, 88, 0, 88]); // effective y = new Float64Array([0, 0]) with incy = 2
      // Symmetric A*x = [1*1+2*2; 2*1+3*2] = [5; 8]
      const expected = new Float64Array([5, 88, 8, 88]);

      dsymv(Triangular.Upper, n, alpha, A, n, x, 2, beta, y, 2);

      expect(y).toEqual(expected);
    });
  });

  describe("DGER - General Rank-1 Update", () => {
    it("should compute A = alpha*x*y^T + A correctly", () => {
      const m = 2,
        n = 3;
      const alpha = 2.0;
      const x = new Float64Array([1, 2]);
      const y = new Float64Array([3, 4, 5]);
      const A = new Float64Array([10, 40, 20, 50, 30, 60]); // 2x3 matrix in column-major: [new Float64Array([10,20,30]),new Float64Array([40,50,60])]
      // alpha*x*y^T = 2*[1; 2]*[3 4 5] = 2*[[3 4 5]; [6 8 10]] = [[6 8 10]; [12 16 20]]
      // A + alpha*x*y^T = [[16 28 40]; [52 66 80]]
      const expected = new Float64Array([16, 52, 28, 66, 40, 80]);

      dger(m, n, alpha, x, 1, y, 1, A, m);

      expect(A).toEqual(expected);
    });

    it("should handle alpha = 0 (no update)", () => {
      const m = 2,
        n = 2;
      const alpha = 0.0;
      const x = new Float64Array([1, 2]);
      const y = new Float64Array([3, 4]);
      const A = new Float64Array([10, 30, 20, 40]); // 2x2 matrix in column-major: [new Float64Array([10,20]),new Float64Array([30,40])]
      const expected = new Float64Array([10, 30, 20, 40]); // No change

      dger(m, n, alpha, x, 1, y, 1, A, m);

      expect(A).toEqual(expected);
    });

    it("should handle zero elements in y vector", () => {
      const m = 2,
        n = 3;
      const alpha = 1.0;
      const x = new Float64Array([1, 2]);
      const y = new Float64Array([0, 4, 0]); // y[0] and y[2] are zero
      const A = new Float64Array([10, 40, 20, 50, 30, 60]); // 2x3 matrix in column-major: [new Float64Array([10,20,30]),new Float64Array([40,50,60])]
      // Only y[1] contributes: A[i]new Float64Array([1]) += x[i] * alpha * y[1] = x[i] * 4
      const expected = new Float64Array([10, 40, 24, 58, 30, 60]);

      dger(m, n, alpha, x, 1, y, 1, A, m);

      expect(A).toEqual(expected);
    });

    it("should handle different increments", () => {
      const m = 2,
        n = 2;
      const alpha = 1.0;
      const x = new Float64Array([1, 99, 2, 99]); // effective x = new Float64Array([1, 2]) with incx = 2
      const y = new Float64Array([3, 88, 4, 88]); // effective y = new Float64Array([3, 4]) with incy = 2
      const A = new Float64Array([10, 30, 20, 40]); // 2x2 matrix in column-major: [new Float64Array([10,20]),new Float64Array([30,40])]
      // x*y^T = [1; 2]*[3 4] = [[3 4]; [6 8]]
      // A + x*y^T = [[13 24]; [36 48]]
      const expected = new Float64Array([13, 36, 24, 48]);

      dger(m, n, alpha, x, 2, y, 2, A, m);

      expect(A).toEqual(expected);
    });
  });

  describe("DTRMV - Triangular Matrix-Vector Multiply", () => {
    it("should compute x := A*x correctly (upper triangle, non-unit)", () => {
      const n = 3;
      const A = new Float64Array([2, 0, 0, 3, 5, 0, 4, 6, 7]); // 3x3 upper triangular matrix in column-major: [new Float64Array([2,3,4]),new Float64Array([0,5,6]),new Float64Array([0,0,7])]
      const x = new Float64Array([1, 2, 3]);
      // A*x for upper triangular: x[0] = 2*1 + 3*2 + 4*3 = 20, x[1] = 5*2 + 6*3 = 28, x[2] = 7*3 = 21
      const expected = new Float64Array([20, 28, 21]);

      dtrmv(
        Triangular.Upper,
        Transpose.NoTranspose,
        Diagonal.NonUnit,
        n,
        A,
        n,
        x,
        1
      );

      expect(x).toEqual(expected);
    });

    it("should compute x := A^T*x correctly (upper triangle)", () => {
      const n = 3;
      const A = new Float64Array([2, 0, 0, 3, 5, 0, 4, 6, 7]); // 3x3 upper triangular matrix in column-major: [new Float64Array([2,3,4]),new Float64Array([0,5,6]),new Float64Array([0,0,7])]
      const x = new Float64Array([1, 2, 3]);
      // A^T*x: [2 0 0; 3 5 0; 4 6 7] * [1; 2; 3] = new Float64Array([2, 13, 37])
      const expected = new Float64Array([2, 13, 37]);

      dtrmv(
        Triangular.Upper,
        Transpose.Transpose,
        Diagonal.NonUnit,
        n,
        A,
        n,
        x,
        1
      );

      expect(x).toEqual(expected);
    });

    it("should handle unit triangular matrices", () => {
      const n = 3;
      const A = new Float64Array([999, 0, 0, 2, 999, 0, 3, 4, 999]); // 3x3 upper triangular matrix in column-major with diagonal ignored for unit triangular
      const x = new Float64Array([1, 2, 3]);
      // Unit diagonal assumed: x[0] = 1 + 2*2 + 3*3 = 14, x[1] = 2 + 4*3 = 14, x[2] = 3
      const expected = new Float64Array([14, 14, 3]);

      dtrmv(
        Triangular.Upper,
        Transpose.NoTranspose,
        Diagonal.Unit,
        n,
        A,
        n,
        x,
        1
      );

      expect(x).toEqual(expected);
    });

    it("should handle lower triangular matrices", () => {
      const n = 3;
      const A = new Float64Array([2, 3, 4, 0, 5, 6, 0, 0, 7]); // 3x3 lower triangular matrix in column-major: [new Float64Array([2,0,0]),new Float64Array([3,5,0]),new Float64Array([4,6,7])]
      const x = new Float64Array([1, 2, 3]);
      // Lower triangular A*x: x[0] = 2*1 = 2, x[1] = 3*1 + 5*2 = 13, x[2] = 4*1 + 6*2 + 7*3 = 37
      const expected = new Float64Array([2, 13, 37]);

      dtrmv(
        Triangular.Lower,
        Transpose.NoTranspose,
        Diagonal.NonUnit,
        n,
        A,
        n,
        x,
        1
      );

      expect(x).toEqual(expected);
    });
  });

  describe("DTRSV - Triangular Solve", () => {
    it("should solve A*x = b correctly (upper triangle)", () => {
      const n = 3;
      const A = new Float64Array([2, 0, 0, 1, 3, 0, 1, 2, 4]); // 3x3 upper triangular matrix in column-major: [new Float64Array([2,1,1]),new Float64Array([0,3,2]),new Float64Array([0,0,4])]
      const x = new Float64Array([6, 11, 12]); // This is the b vector, will be overwritten with solution
      // Solving A*x = b: 2*x[0] + 1*x[1] + 1*x[2] = 6, 3*x[1] + 2*x[2] = 11, 4*x[2] = 12
      // Back substitution: x[2] = 3, x[1] = (11 - 2*3)/3 = 5/3, x[0] = (6 - 5/3 - 3)/2 = 2/3
      const expected = [2 / 3, 5 / 3, 3];

      dtrsv(
        Triangular.Upper,
        Transpose.NoTranspose,
        Diagonal.NonUnit,
        n,
        A,
        n,
        x,
        1
      );

      for (let i = 0; i < n; i++) {
        expect(x[i]).toBeCloseTo(expected[i], 10);
      }
    });

    it("should solve A^T*x = b correctly (upper triangle)", () => {
      const n = 2;
      const A = new Float64Array([2, 0, 3, 4]); // 2x2 matrix in column-major: [new Float64Array([2,3]),new Float64Array([0,4])]
      const x = new Float64Array([7, 4]); // b vector
      // A^T = [new Float64Array([2, 0]), new Float64Array([3, 4])], solving: 2*x[0] = 7 => x[0] = 3.5, 3*3.5 + 4*x[1] = 4 => x[1] = -1.625
      const expected = new Float64Array([3.5, -1.625]);

      dtrsv(
        Triangular.Upper,
        Transpose.Transpose,
        Diagonal.NonUnit,
        n,
        A,
        n,
        x,
        1
      );

      for (let i = 0; i < n; i++) {
        expect(x[i]).toBeCloseTo(expected[i], 10);
      }
    });

    it("should handle unit triangular matrices", () => {
      const n = 2;
      const A = new Float64Array([999, 0, 2, 999]); // 2x2 matrix in column-major: [new Float64Array([999,2]),new Float64Array([0,999])] - diagonal ignored
      const x = new Float64Array([5, 3]); // b vector
      // Unit diagonal: x[0] + 2*x[1] = 5, x[1] = 3 => x[0] = 5 - 2*3 = -1
      const expected = new Float64Array([-1, 3]);

      dtrsv(
        Triangular.Upper,
        Transpose.NoTranspose,
        Diagonal.Unit,
        n,
        A,
        n,
        x,
        1
      );

      expect(x).toEqual(expected);
    });

    it("should handle lower triangular matrices", () => {
      const n = 2;
      const A = new Float64Array([3, 2, 0, 4]); // 2x2 matrix in column-major: [new Float64Array([3,0]),new Float64Array([2,4])]
      const x = new Float64Array([6, 14]); // b vector
      // Forward substitution: 3*x[0] = 6 => x[0] = 2, 2*2 + 4*x[1] = 14 => x[1] = 2.5
      const expected = new Float64Array([2, 2.5]);

      dtrsv(
        Triangular.Lower,
        Transpose.NoTranspose,
        Diagonal.NonUnit,
        n,
        A,
        n,
        x,
        1
      );

      expect(x).toEqual(expected);
    });
  });

  describe("DSYR - Symmetric Rank-1 Update", () => {
    it("should compute A := alpha*x*x^T + A correctly (upper triangle)", () => {
      const n = 2;
      const alpha = 2.0;
      const x = new Float64Array([1, 3]);
      const A = new Float64Array([5, 0, 7, 9]); // 2x2 matrix in column-major: [new Float64Array([5,7]),new Float64Array([0,9])] - lower triangle not referenced
      // alpha*x*x^T = 2*[1; 3]*[1 3] = 2*[[1 3]; [3 9]] = [[2 6]; [6 18]]
      // A + alpha*x*x^T for upper triangle: [[5+2 7+6]; [0 9+18]] = [[7 13]; [0 27]]
      const expected = new Float64Array([7, 0, 13, 27]);

      dsyr(Triangular.Upper, n, alpha, x, 1, A, n);

      expect(A).toEqual(expected);
    });

    it("should compute A := alpha*x*x^T + A correctly (lower triangle)", () => {
      const n = 2;
      const alpha = 1.0;
      const x = new Float64Array([2, 1]);
      const A = new Float64Array([3, 5, 0, 7]); // 2x2 matrix in column-major: [new Float64Array([3,0]),new Float64Array([5,7])] - upper triangle not referenced
      // alpha*x*x^T = [2; 1]*[2 1] = [[4 2]; [2 1]]
      // A + alpha*x*x^T for lower triangle: [[3+4 0]; [5+2 7+1]] = [[7 0]; [7 8]]
      const expected = new Float64Array([7, 7, 0, 8]);

      dsyr(Triangular.Lower, n, alpha, x, 1, A, n);

      expect(A).toEqual(expected);
    });

    it("should handle alpha = 0 (no update)", () => {
      const n = 2;
      const alpha = 0.0;
      const x = new Float64Array([5, 10]);
      const A = new Float64Array([1, 0, 2, 3]); // 2x2 matrix in column-major: [new Float64Array([1,2]),new Float64Array([0,3])]
      const expected = new Float64Array([1, 0, 2, 3]); // No change

      dsyr(Triangular.Upper, n, alpha, x, 1, A, n);

      expect(A).toEqual(expected);
    });

    it("should handle zero elements in x vector", () => {
      const n = 3;
      const alpha = 1.0;
      const x = new Float64Array([0, 2, 0]); // Only x[1] is non-zero
      const A = new Float64Array([1, 0, 0, 2, 5, 6, 3, 6, 9]); // 3x3 matrix in column-major: [new Float64Array([1,2,3]),new Float64Array([0,5,6]),new Float64Array([0,6,9])]
      // Only x[1] contributes: A[1]new Float64Array([1]) += x[1]*alpha*x[1] = 5 + 2*1*2 = 9
      const expected = new Float64Array([1, 0, 0, 2, 9, 6, 3, 6, 9]);

      dsyr(Triangular.Upper, n, alpha, x, 1, A, n);

      expect(A).toEqual(expected);
    });

    it("should handle different increments", () => {
      const n = 2;
      const alpha = 1.0;
      const x = new Float64Array([1, 99, 2, 99]); // effective x = new Float64Array([1, 2]) with incx = 2
      const A = new Float64Array([10, 30, 20, 40]); // 2x2 matrix in column-major: [new Float64Array([10,20]),new Float64Array([30,40])]
      // x*x^T = [1; 2]*[1 2] = [[1 2]; [2 4]] for lower triangle
      // A + x*x^T = [[11 20]; [32 44]]
      const expected = new Float64Array([11, 32, 20, 44]);

      dsyr(Triangular.Lower, n, alpha, x, 2, A, n);

      expect(A).toEqual(expected);
    });
  });
});
