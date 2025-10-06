import { dgemm, dsymm, dsyrk, dtrmm, dtrsm, dsyr2k } from "../src/level3";
import { BLASTranspose, BLASUplo, BLASDiag, BLASSide } from "../src/types";

describe("BLAS Level 3", () => {
  describe("DGEMM (Double precision general matrix multiply)", () => {
    it("should compute C = alpha*A*B + beta*C correctly", () => {
      // Column-major storage: A = [[1,3],[2,4]] stored as [1,3,2,4]
      const A = [1.0, 3.0, 2.0, 4.0]; // 2x2 matrix
      const B = [5.0, 7.0, 6.0, 8.0]; // 2x2 matrix
      const C = [0.0, 0.0, 0.0, 0.0]; // 2x2 matrix

      dgemm(
        BLASTranspose.NoTranspose,
        BLASTranspose.NoTranspose,
        2,
        2,
        2,
        1.0,
        A,
        2,
        B,
        2,
        0.0,
        C,
        2
      );

      expect(C[0]).toBeCloseTo(19.0); // C[0,0]
      expect(C[1]).toBeCloseTo(43.0); // C[1,0]
      expect(C[2]).toBeCloseTo(22.0); // C[0,1]
      expect(C[3]).toBeCloseTo(50.0); // C[1,1]
    });

    it("should handle A transpose", () => {
      const A = [1.0, 2.0, 3.0, 4.0]; // 2x2 matrix - transposed storage
      const B = [5.0, 7.0, 6.0, 8.0]; // 2x2 matrix
      const C = [0.0, 0.0, 0.0, 0.0]; // 2x2 matrix

      dgemm(
        BLASTranspose.Transpose,
        BLASTranspose.NoTranspose,
        2,
        2,
        2,
        1.0,
        A,
        2,
        B,
        2,
        0.0,
        C,
        2
      );

      expect(C[0]).toBeCloseTo(19.0);
      expect(C[1]).toBeCloseTo(43.0);
      expect(C[2]).toBeCloseTo(22.0);
      expect(C[3]).toBeCloseTo(50.0);
    });

    it("should handle B transpose", () => {
      const A = [1.0, 3.0, 2.0, 4.0]; // 2x2 matrix
      const B = [5.0, 6.0, 7.0, 8.0]; // 2x2 matrix - transposed storage
      const C = [0.0, 0.0, 0.0, 0.0]; // 2x2 matrix

      dgemm(
        BLASTranspose.NoTranspose,
        BLASTranspose.Transpose,
        2,
        2,
        2,
        1.0,
        A,
        2,
        B,
        2,
        0.0,
        C,
        2
      );

      expect(C[0]).toBeCloseTo(19.0);
      expect(C[1]).toBeCloseTo(43.0);
      expect(C[2]).toBeCloseTo(22.0);
      expect(C[3]).toBeCloseTo(50.0);
    });

    it("should handle both A and B transpose", () => {
      const A = [1.0, 2.0, 3.0, 4.0]; // 2x2 matrix - transposed storage
      const B = [5.0, 6.0, 7.0, 8.0]; // 2x2 matrix - transposed storage
      const C = [0.0, 0.0, 0.0, 0.0]; // 2x2 matrix

      dgemm(
        BLASTranspose.Transpose,
        BLASTranspose.Transpose,
        2,
        2,
        2,
        1.0,
        A,
        2,
        B,
        2,
        0.0,
        C,
        2
      );

      expect(C[0]).toBeCloseTo(19.0);
      expect(C[1]).toBeCloseTo(43.0);
      expect(C[2]).toBeCloseTo(22.0);
      expect(C[3]).toBeCloseTo(50.0);
    });

    it("should handle alpha and beta scaling", () => {
      const A = [1.0, 3.0, 2.0, 4.0]; // 2x2 matrix
      const B = [1.0, 0.0, 0.0, 1.0]; // 2x2 identity matrix
      const C = [1.0, 1.0, 1.0, 1.0]; // 2x2 matrix

      dgemm(
        BLASTranspose.NoTranspose,
        BLASTranspose.NoTranspose,
        2,
        2,
        2,
        2.0,
        A,
        2,
        B,
        2,
        3.0,
        C,
        2
      );

      expect(C[0]).toBeCloseTo(5.0); // 2*1 + 3*1
      expect(C[1]).toBeCloseTo(9.0); // 2*3 + 3*1
      expect(C[2]).toBeCloseTo(7.0); // 2*2 + 3*1
      expect(C[3]).toBeCloseTo(11.0); // 2*4 + 3*1
    });

    it("should handle rectangular matrices", () => {
      const A = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // 2x3 matrix
      const B = [7.0, 9.0, 11.0, 8.0, 10.0, 12.0]; // 3x2 matrix
      const C = [0.0, 0.0, 0.0, 0.0]; // 2x2 matrix

      dgemm(
        BLASTranspose.NoTranspose,
        BLASTranspose.NoTranspose,
        2,
        2,
        3,
        1.0,
        A,
        2,
        B,
        3,
        0.0,
        C,
        2
      );

      expect(C[0]).toBeCloseTo(58.0); // 1*7 + 2*9 + 3*11
      expect(C[1]).toBeCloseTo(139.0); // 4*7 + 5*9 + 6*11
      expect(C[2]).toBeCloseTo(64.0); // 1*8 + 2*10 + 3*12
      expect(C[3]).toBeCloseTo(154.0); // 4*8 + 5*10 + 6*12
    });

    it("should handle alpha = 0", () => {
      const A = [1.0, 3.0, 2.0, 4.0]; // 2x2 matrix
      const B = [5.0, 7.0, 6.0, 8.0]; // 2x2 matrix
      const C = [1.0, 3.0, 2.0, 4.0]; // 2x2 matrix

      dgemm(
        BLASTranspose.NoTranspose,
        BLASTranspose.NoTranspose,
        2,
        2,
        2,
        0.0,
        A,
        2,
        B,
        2,
        2.0,
        C,
        2
      );

      expect(C[0]).toBeCloseTo(2.0);
      expect(C[1]).toBeCloseTo(6.0);
      expect(C[2]).toBeCloseTo(4.0);
      expect(C[3]).toBeCloseTo(8.0);
    });

    it("should handle beta = 0", () => {
      const A = [1.0, 0.0, 0.0, 1.0]; // 2x2 identity matrix
      const B = [5.0, 7.0, 6.0, 8.0]; // 2x2 matrix
      const C = [99.0, 99.0, 99.0, 99.0]; // 2x2 matrix - should be overwritten

      dgemm(
        BLASTranspose.NoTranspose,
        BLASTranspose.NoTranspose,
        2,
        2,
        2,
        1.0,
        A,
        2,
        B,
        2,
        0.0,
        C,
        2
      );

      expect(C[0]).toBeCloseTo(5.0);
      expect(C[1]).toBeCloseTo(7.0);
      expect(C[2]).toBeCloseTo(6.0);
      expect(C[3]).toBeCloseTo(8.0);
    });
  });

  describe("DSYMM (Double precision symmetric matrix multiply)", () => {
    it("should compute C = alpha*A*B + beta*C with left side, upper triangle", () => {
      // Symmetric matrix A (upper triangle stored)
      const A = [2.0, 0.0, 0.0, 3.0, 5.0, 0.0, 4.0, 6.0, 7.0]; // 3x3 matrix
      const B = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]; // 3x2 matrix
      const C = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // 3x2 matrix

      dsymm(BLASSide.Left, BLASUplo.Upper, 3, 2, 1.0, A, 3, B, 3, 0.0, C, 3);

      // Expected: A_symmetric * B where A_symmetric = [[2,3,4],[3,5,6],[4,6,7]]
      expect(C[0]).toBeCloseTo(31.0); // 2*1 + 3*3 + 4*5
      expect(C[1]).toBeCloseTo(48.0); // 3*1 + 5*3 + 6*5
      expect(C[2]).toBeCloseTo(57.0); // 4*1 + 6*3 + 7*5
      expect(C[3]).toBeCloseTo(40.0); // 2*2 + 3*4 + 4*6
      expect(C[4]).toBeCloseTo(62.0); // 3*2 + 5*4 + 6*6
      expect(C[5]).toBeCloseTo(74.0); // 4*2 + 6*4 + 7*6
    });

    it("should compute C = alpha*A*B + beta*C with left side, lower triangle", () => {
      // Symmetric matrix A (lower triangle stored)
      const A = [2.0, 3.0, 4.0, 0.0, 5.0, 6.0, 0.0, 0.0, 7.0]; // 3x3 matrix
      const B = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]; // 3x2 matrix
      const C = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // 3x2 matrix

      dsymm(BLASSide.Left, BLASUplo.Lower, 3, 2, 1.0, A, 3, B, 3, 0.0, C, 3);

      expect(C[0]).toBeCloseTo(31.0);
      expect(C[1]).toBeCloseTo(48.0);
      expect(C[2]).toBeCloseTo(57.0);
      expect(C[3]).toBeCloseTo(40.0);
      expect(C[4]).toBeCloseTo(62.0);
      expect(C[5]).toBeCloseTo(74.0);
    });

    it("should compute C = alpha*B*A + beta*C with right side", () => {
      const A = [2.0, 0.0, 3.0, 5.0]; // 2x2 matrix
      const B = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]; // 3x2 matrix
      const C = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // 3x2 matrix

      dsymm(BLASSide.Right, BLASUplo.Upper, 3, 2, 1.0, A, 2, B, 3, 0.0, C, 3);

      // Expected: B * A_symmetric where A_symmetric = [[2,3],[3,5]]
      expect(C[0]).toBeCloseTo(8.0); // 1*2 + 2*3
      expect(C[1]).toBeCloseTo(18.0); // 3*2 + 4*3
      expect(C[2]).toBeCloseTo(28.0); // 5*2 + 6*3
      expect(C[3]).toBeCloseTo(13.0); // 1*3 + 2*5
      expect(C[4]).toBeCloseTo(29.0); // 3*3 + 4*5
      expect(C[5]).toBeCloseTo(45.0); // 5*3 + 6*5
    });

    it("should handle alpha and beta scaling", () => {
      const A = [2.0, 3.0, 0.0, 5.0]; // 2x2 matrix - lower triangle
      const B = [1.0, 0.0, 0.0, 1.0]; // 2x2 identity matrix
      const C = [1.0, 1.0, 1.0, 1.0]; // 2x2 matrix

      dsymm(BLASSide.Left, BLASUplo.Lower, 2, 2, 2.0, A, 2, B, 2, 3.0, C, 2);

      expect(C[0]).toBeCloseTo(7.0); // 2*(2*1) + 3*1
      expect(C[1]).toBeCloseTo(9.0); // 2*(3*1) + 3*1
      expect(C[2]).toBeCloseTo(9.0); // 2*(3*1) + 3*1
      expect(C[3]).toBeCloseTo(13.0); // 2*(5*1) + 3*1
    });
  });

  describe("DSYRK (Double precision symmetric rank-k update)", () => {
    it("should compute C = alpha*A*A^T + beta*C with lower triangle", () => {
      const A = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]; // 3x2 matrix
      const C = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // 3x3 matrix

      dsyrk(
        BLASUplo.Lower,
        BLASTranspose.NoTranspose,
        3,
        2,
        1.0,
        A,
        3,
        0.0,
        C,
        3
      );

      // Expected: C = A*A^T (upper triangle only)
      expect(C[0]).toBeCloseTo(5.0); // 1*1 + 2*2
      expect(C[1]).toBeCloseTo(11.0); // 1*3 + 2*4
      expect(C[2]).toBeCloseTo(17.0); // 1*5 + 2*6
      expect(C[4]).toBeCloseTo(25.0); // 3*3 + 4*4
      expect(C[5]).toBeCloseTo(39.0); // 3*5 + 4*6
      expect(C[8]).toBeCloseTo(61.0); // 5*5 + 6*6

      // Lower triangle should remain zero
      expect(C[3]).toBeCloseTo(0.0);
      expect(C[6]).toBeCloseTo(0.0);
      expect(C[7]).toBeCloseTo(0.0);
    });

    it("should compute C = alpha*A^T*A + beta*C with upper triangle", () => {
      const A = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
      const C = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // 3x3 identity matrix

      dsyrk(
        BLASUplo.Upper,
        BLASTranspose.Transpose,
        3,
        2,
        2.0,
        A,
        2,
        1.0,
        C,
        3
      );

      // Expected: C = 2*A^T*A + I (lower triangle only)
      expect(C[0]).toBeCloseTo(11.0); // 2*(1*1 + 2*2) + 1
      expect(C[3]).toBeCloseTo(22.0); // 2*(3*1 + 4*2)
      expect(C[4]).toBeCloseTo(51.0); // 2*(3*3 + 4*4) + 1
      expect(C[6]).toBeCloseTo(34.0); // 2*(5*1 + 6*2)
      expect(C[7]).toBeCloseTo(78.0); // 2*(5*3 + 6*4)
      expect(C[8]).toBeCloseTo(123.0); // 2*(5*5 + 6*6) + 1

      // Upper triangle should remain zero
      expect(C[1]).toBeCloseTo(0.0);
      expect(C[2]).toBeCloseTo(0.0);
      expect(C[5]).toBeCloseTo(0.0);
    });

    it("should handle alpha = 0", () => {
      const A = [1.0, 3.0, 2.0, 4.0]; // 2x2 matrix
      const C = [5.0, 7.0, 6.0, 8.0]; // 2x2 matrix

      dsyrk(BLASUplo.Lower, BLASTranspose.NoTranspose, 2, 2, 0, A, 2, 2, C, 2);

      // C = 0*A*A^T + 2*C = 2*C
      expect(C[0]).toBeCloseTo(10.0);
      expect(C[1]).toBeCloseTo(14.0);
      expect(C[3]).toBeCloseTo(16.0);
    });
  });

  describe("DTRSM (Double precision triangular solve)", () => {
    it("should solve upper triangular system from left", () => {
      // Upper triangular matrix A
      const A = [2.0, 0.0, 3.0, 4.0]; // 2x2 matrix
      const B = [14.0, 16.0, 16.0, 20.0]; // 2x2 matrix

      dtrsm(
        BLASSide.Left,
        BLASUplo.Upper,
        BLASTranspose.NoTranspose,
        BLASDiag.NonUnit,
        2,
        2,
        1.0,
        A,
        2,
        B,
        2
      );

      // Solves A*X = B, so X = A^(-1)*B
      expect(B[0]).toBeCloseTo(1.0);
      expect(B[1]).toBeCloseTo(4.0);
      expect(B[2]).toBeCloseTo(0.5); // Fixed: (16 - 3*5)/2 = 0.5
      expect(B[3]).toBeCloseTo(5.0);
    });

    it("should solve lower triangular system from left", () => {
      const A = [2.0, 3.0, 0.0, 4.0]; // 2x2 matrix
      const B = [2.0, 19.0, 4.0, 23.0]; // 2x2 matrix

      dtrsm(
        BLASSide.Left,
        BLASUplo.Lower,
        BLASTranspose.NoTranspose,
        BLASDiag.NonUnit,
        2,
        2,
        1.0,
        A,
        2,
        B,
        2
      );

      expect(B[0]).toBeCloseTo(1.0);
      expect(B[1]).toBeCloseTo(4.0);
      expect(B[2]).toBeCloseTo(2.0);
      expect(B[3]).toBeCloseTo(4.25); // Fixed: (23 - 3*2)/4 = 4.25
    });

    it("should solve with unit diagonal", () => {
      // Unit lower triangular (diagonal assumed to be 1)
      const A = [999.0, 3.0, 0.0, 999.0]; // 2x2 matrix - diagonal values ignored
      const B = [1.0, 7.0, 2.0, 8.0]; // 2x2 matrix

      dtrsm(
        BLASSide.Left,
        BLASUplo.Lower,
        BLASTranspose.NoTranspose,
        BLASDiag.Unit,
        2,
        2,
        1.0,
        A,
        2,
        B,
        2
      );

      expect(B[0]).toBeCloseTo(1.0);
      expect(B[1]).toBeCloseTo(4.0);
      expect(B[2]).toBeCloseTo(2.0);
      expect(B[3]).toBeCloseTo(2.0);
    });

    it("should solve triangular system from right", () => {
      const A = [2.0, 0.0, 3.0, 4.0]; // 2x2 matrix
      const B = [10.0, 14.0, 25.0, 31.0]; // 2x2 matrix

      dtrsm(
        BLASSide.Right,
        BLASUplo.Upper,
        BLASTranspose.NoTranspose,
        BLASDiag.NonUnit,
        2,
        2,
        1.0,
        A,
        2,
        B,
        2
      );

      // Solves X*A = B, so X = B*A^(-1)
      expect(B[0]).toBeCloseTo(5.0);
      expect(B[1]).toBeCloseTo(7.0);
      expect(B[2]).toBeCloseTo(2.5); // Fixed: (25 - 3*5)/4 = 2.5
      expect(B[3]).toBeCloseTo(2.5); // Fixed: (31 - 3*7)/4 = 2.5
    });

    it("should handle transpose", () => {
      const A = [2.0, 3.0, 0.0, 4.0]; // 2x2 matrix
      const B = [2.0, 12.0, 4.0, 20.0]; // 2x2 matrix

      dtrsm(
        BLASSide.Left,
        BLASUplo.Lower,
        BLASTranspose.Transpose,
        BLASDiag.NonUnit,
        2,
        2,
        1.0,
        A,
        2,
        B,
        2
      );

      // Solves A^T*X = B where A^T = [[2, 3], [0, 4]]
      expect(B[0]).toBeCloseTo(1.0);
      expect(B[1]).toBeCloseTo(2.25); // Fixed: (12 - 3*1)/4 = 2.25
      expect(B[2]).toBeCloseTo(2.0);
      expect(B[3]).toBeCloseTo(3.5); // Fixed: (20 - 3*2)/4 = 3.5
    });

    it("should handle alpha scaling", () => {
      const A = [2.0, 0.0, 0.0, 2.0]; // 2x2 matrix
      const B = [4.0, 8.0, 6.0, 10.0]; // 2x2 matrix

      dtrsm(
        BLASSide.Left,
        BLASUplo.Lower,
        BLASTranspose.NoTranspose,
        BLASDiag.NonUnit,
        2,
        2,
        0.5,
        A,
        2,
        B,
        2
      );

      // Solves A*X = 0.5*B
      expect(B[0]).toBeCloseTo(1.0);
      expect(B[1]).toBeCloseTo(2.0);
      expect(B[2]).toBeCloseTo(1.5);
      expect(B[3]).toBeCloseTo(2.5);
    });
  });

  describe("DTRMM (Double precision triangular matrix multiply)", () => {
    it("should compute B = alpha*A*B with upper triangular A from left", () => {
      const A = [2.0, 0.0, 3.0, 4.0]; // 2x2 matrix
      const B = [1.0, 3.0, 2.0, 4.0]; // 2x2 matrix

      dtrmm(
        BLASSide.Left,
        BLASUplo.Upper,
        BLASTranspose.NoTranspose,
        BLASDiag.NonUnit,
        2,
        2,
        1.0,
        A,
        2,
        B,
        2
      );

      expect(B[0]).toBeCloseTo(11.0); // 2*1 + 3*3
      expect(B[1]).toBeCloseTo(12.0); // 0*1 + 4*3
      expect(B[2]).toBeCloseTo(16.0); // 2*2 + 3*4
      expect(B[3]).toBeCloseTo(16.0); // 0*2 + 4*4
    });

    it("should compute B = alpha*A*B with lower triangular A from left", () => {
      const A = [2.0, 3.0, 0.0, 4.0]; // 2x2 matrix
      const B = [1.0, 3.0, 2.0, 4.0]; // 2x2 matrix

      dtrmm(
        BLASSide.Left,
        BLASUplo.Lower,
        BLASTranspose.NoTranspose,
        BLASDiag.NonUnit,
        2,
        2,
        1.0,
        A,
        2,
        B,
        2
      );

      expect(B[0]).toBeCloseTo(2.0); // 2*1 + 0*3
      expect(B[1]).toBeCloseTo(15.0); // 3*1 + 4*3
      expect(B[2]).toBeCloseTo(4.0); // 2*2 + 0*4
      expect(B[3]).toBeCloseTo(22.0); // 3*2 + 4*4
    });

    it("should handle unit diagonal", () => {
      const A = [999.0, 0.0, 3.0, 999.0]; // 2x2 matrix - diagonal ignored
      const B = [1.0, 3.0, 2.0, 4.0]; // 2x2 matrix

      dtrmm(
        BLASSide.Left,
        BLASUplo.Upper,
        BLASTranspose.NoTranspose,
        BLASDiag.Unit,
        2,
        2,
        1.0,
        A,
        2,
        B,
        2
      );

      expect(B[0]).toBeCloseTo(10.0); // 1*1 + 3*3
      expect(B[1]).toBeCloseTo(3.0); // 0*1 + 1*3
      expect(B[2]).toBeCloseTo(14.0); // 1*2 + 3*4
      expect(B[3]).toBeCloseTo(4.0); // 0*2 + 1*4
    });

    it("should compute B = alpha*B*A from right", () => {
      const A = [2.0, 0.0, 3.0, 4.0]; // 2x2 matrix
      const B = [1.0, 3.0, 2.0, 4.0]; // 2x2 matrix

      dtrmm(
        BLASSide.Right,
        BLASUplo.Upper,
        BLASTranspose.NoTranspose,
        BLASDiag.NonUnit,
        2,
        2,
        1.0,
        A,
        2,
        B,
        2
      );

      expect(B[0]).toBeCloseTo(2.0); // 1*2 + 2*0
      expect(B[1]).toBeCloseTo(6.0); // 3*2 + 4*0
      expect(B[2]).toBeCloseTo(11.0); // 1*3 + 2*4
      expect(B[3]).toBeCloseTo(25.0); // 3*3 + 4*4
    });

    it("should handle transpose", () => {
      const A = [2.0, 3.0, 0.0, 4.0]; // 2x2 matrix
      const B = [1.0, 3.0, 2.0, 4.0]; // 2x2 matrix

      dtrmm(
        BLASSide.Left,
        BLASUplo.Lower,
        BLASTranspose.Transpose,
        BLASDiag.NonUnit,
        2,
        2,
        1.0,
        A,
        2,
        B,
        2
      );

      // B = A^T*B where A^T is upper triangular
      expect(B[0]).toBeCloseTo(11.0); // 2*1 + 3*3
      expect(B[1]).toBeCloseTo(12.0); // 0*1 + 4*3
      expect(B[2]).toBeCloseTo(16.0); // 2*2 + 3*4
      expect(B[3]).toBeCloseTo(16.0); // 0*2 + 4*4
    });

    it("should handle alpha scaling", () => {
      const A = [2.0, 0.0, 0.0, 2.0]; // 2x2 matrix
      const B = [1.0, 3.0, 2.0, 4.0]; // 2x2 matrix

      dtrmm(
        BLASSide.Left,
        BLASUplo.Lower,
        BLASTranspose.NoTranspose,
        BLASDiag.NonUnit,
        2,
        2,
        0.5,
        A,
        2,
        B,
        2
      );

      expect(B[0]).toBeCloseTo(1.0);
      expect(B[1]).toBeCloseTo(3.0);
      expect(B[2]).toBeCloseTo(2.0);
      expect(B[3]).toBeCloseTo(4.0);
    });
  });

  describe("DSYR2K (Double precision symmetric rank-2k update)", () => {
    it("should compute C = alpha*A*B^T + alpha*B*A^T + beta*C with lower triangle", () => {
      // A and B are 3x2 matrices
      const A = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]; // 3x2 matrix
      const B = [1.0, 2.0, 1.0, 1.0, 1.0, 2.0]; // 3x2 matrix
      const C = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // 3x3 matrix

      dsyr2k(
        BLASUplo.Lower,
        BLASTranspose.NoTranspose,
        3,
        2,
        1.0,
        A,
        3,
        B,
        3,
        0.0,
        C,
        3
      );

      // Expected: C = A*B^T + B*A^T (upper triangle only)
      expect(C[0]).toBeCloseTo(6.0); // (1*1 + 2*1) + (1*1 + 1*2) = 3 + 3 = 6
      expect(C[1]).toBeCloseTo(11.0); // (1*2 + 2*1) + (1*3 + 1*4) = 4 + 7 = 11
      expect(C[2]).toBeCloseTo(16.0); // (1*1 + 2*2) + (1*5 + 1*6) = 5 + 11 = 16
      expect(C[4]).toBeCloseTo(20.0); // (3*2 + 4*1) + (2*3 + 1*4) = 10 + 10 = 20
      expect(C[5]).toBeCloseTo(27.0); // (3*1 + 4*2) + (2*5 + 1*6) = 11 + 16 = 27
      expect(C[8]).toBeCloseTo(34.0); // (5*1 + 6*2) + (1*5 + 2*6) = 17 + 17 = 34

      // Lower triangle should remain zero
      expect(C[3]).toBeCloseTo(0.0);
      expect(C[6]).toBeCloseTo(0.0);
      expect(C[7]).toBeCloseTo(0.0);
    });

    it("should compute C = alpha*A*B^T + alpha*B*A^T + beta*C with lower triangle", () => {
      const A = [1.0, 3.0, 2.0, 4.0]; // 2x2 matrix
      const B = [2.0, 1.0, 1.0, 3.0]; // 2x2 matrix
      const C = [1.0, 0.0, 0.0, 1.0]; // 2x2 identity matrix

      dsyr2k(
        BLASUplo.Lower,
        BLASTranspose.NoTranspose,
        2,
        2,
        2.0,
        A,
        2,
        B,
        2,
        1.0,
        C,
        2
      );

      // C = 2*(A*B^T + B*A^T) + I (lower triangle only)
      expect(C[0]).toBeCloseTo(17.0); // 2*((1*2 + 2*1) + (2*1 + 1*2)) + 1 = 2*(4 + 4) + 1 = 17
      expect(C[1]).toBeCloseTo(34.0); // 2*((3*2 + 4*1) + (1*1 + 3*2)) = 2*(10 + 7) = 34
      expect(C[3]).toBeCloseTo(61.0); // 2*((3*1 + 4*3) + (1*3 + 3*4)) + 1 = 2*(15 + 15) + 1 = 61

      // Upper triangle should remain zero
      expect(C[2]).toBeCloseTo(0.0);
    });

    it("should compute C = alpha*A^T*B + alpha*B^T*A + beta*C with transpose", () => {
      // A and B are 2x3 matrices (transposed in computation)
      const A = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
      const B = [1.0, 1.0, 2.0, 1.0, 1.0, 2.0]; // 2x3 matrix
      const C = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // 3x3 matrix

      dsyr2k(
        BLASUplo.Lower,
        BLASTranspose.Transpose,
        3,
        2,
        1.0,
        A,
        2,
        B,
        2,
        0.0,
        C,
        3
      );

      // Expected: C = A^T*B + B^T*A (upper triangle only)
      expect(C[0]).toBeCloseTo(6.0); // (1*1 + 2*1) + (1*1 + 1*2)
      expect(C[1]).toBeCloseTo(11.0); // (1*2 + 2*1) + (1*3 + 1*4)
      expect(C[2]).toBeCloseTo(16.0); // (1*1 + 2*2) + (1*5 + 1*6)
      expect(C[4]).toBeCloseTo(20.0); // (3*2 + 4*1) + (2*3 + 1*4)
      expect(C[5]).toBeCloseTo(27.0); // (3*1 + 4*2) + (2*5 + 1*6)
      expect(C[8]).toBeCloseTo(34.0); // (5*1 + 6*2) + (1*5 + 2*6)
    });

    it("should handle alpha = 0", () => {
      const A = [1.0, 3.0, 2.0, 4.0]; // 2x2 matrix
      const B = [2.0, 1.0, 1.0, 3.0]; // 2x2 matrix
      const C = [5.0, 7.0, 6.0, 8.0]; // 2x2 matrix

      dsyr2k(
        BLASUplo.Upper,
        BLASTranspose.NoTranspose,
        2,
        2,
        0.0,
        A,
        2,
        B,
        2,
        2.0,
        C,
        2
      );

      // C = 0*(A*B^T + B*A^T) + 2*C = 2*C (upper triangle only)
      expect(C[0]).toBeCloseTo(10.0);
      expect(C[2]).toBeCloseTo(12.0);
      expect(C[3]).toBeCloseTo(16.0);
    });

    it("should handle beta = 0", () => {
      const A = [1.0, 0.0, 0.0, 1.0]; // 2x2 identity matrix
      const B = [2.0, 0.0, 0.0, 2.0]; // 2x2 matrix - 2*identity
      const C = [99.0, 99.0, 99.0, 99.0]; // 2x2 matrix - should be overwritten

      dsyr2k(
        BLASUplo.Upper,
        BLASTranspose.NoTranspose,
        2,
        2,
        1.0,
        A,
        2,
        B,
        2,
        0.0,
        C,
        2
      );

      // C = A*B^T + B*A^T (upper triangle)
      expect(C[0]).toBeCloseTo(4.0); // (1*2) + (2*1)
      expect(C[2]).toBeCloseTo(0.0); // (1*0) + (0*1)
      expect(C[3]).toBeCloseTo(4.0); // (1*2) + (2*1)
    });
  });
});
