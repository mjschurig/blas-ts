import {
  daxpy,
  ddot,
  dcopy,
  dscal,
  drot,
  dswap,
  dnrm2,
  dasum,
  idamax,
  drotg,
} from "../src/level1";

describe("BLAS Level 1 - Vector Operations", () => {
  describe("DAXPY - Vector Scale and Add", () => {
    it("should compute y = alpha*x + y correctly (basic case)", () => {
      const n = 4;
      const alpha = 2.0;
      const x = [1, 2, 3, 4];
      const y = [10, 20, 30, 40];
      const expected = [12, 24, 36, 48]; // y + alpha*x = [10,20,30,40] + 2*[1,2,3,4]

      daxpy(n, alpha, x, 1, y, 1);

      expect(y).toEqual(expected);
    });

    it("should handle alpha = 0 (early return)", () => {
      const n = 3;
      const alpha = 0.0;
      const x = [1, 2, 3];
      const y = [10, 20, 30];
      const expected = [10, 20, 30]; // y should remain unchanged

      daxpy(n, alpha, x, 1, y, 1);

      expect(y).toEqual(expected);
    });

    it("should handle n <= 0 (early return)", () => {
      const n = 0;
      const alpha = 2.0;
      const x = [1, 2, 3];
      const y = [10, 20, 30];
      const expected = [10, 20, 30]; // y should remain unchanged

      daxpy(n, alpha, x, 1, y, 1);

      expect(y).toEqual(expected);
    });

    it("should handle different increments", () => {
      const n = 2;
      const alpha = 3.0;
      const x = [1, 0, 2, 0]; // effective x = [1, 2] with incx = 2
      const y = [10, 0, 20, 0]; // effective y = [10, 20] with incy = 2
      const expected = [13, 0, 26, 0]; // y[0] = 10 + 3*1 = 13, y[2] = 20 + 3*2 = 26

      daxpy(n, alpha, x, 2, y, 2);

      expect(y).toEqual(expected);
    });

    it("should handle negative increments", () => {
      const n = 2;
      const alpha = 2.0;
      const x = [1, 2]; // Will be accessed in reverse due to negative increment
      const y = [10, 20];

      daxpy(n, alpha, x, -1, y, -1);

      // With negative increments, access pattern is reversed
      // First iteration: y[1] = y[1] + alpha * x[1] = 20 + 2*2 = 24
      // Second iteration: y[0] = y[0] + alpha * x[0] = 10 + 2*1 = 12
      expect(y).toEqual([12, 24]);
    });

    it("should handle unrolled loop optimization (n >= 4)", () => {
      const n = 8;
      const alpha = 0.5;
      const x = [1, 2, 3, 4, 5, 6, 7, 8];
      const y = [10, 20, 30, 40, 50, 60, 70, 80];
      const expected = [10.5, 21, 31.5, 42, 52.5, 63, 73.5, 84];

      daxpy(n, alpha, x, 1, y, 1);

      expect(y).toEqual(expected);
    });
  });

  describe("DDOT - Dot Product", () => {
    it("should compute dot product correctly (basic case)", () => {
      const n = 4;
      const x = [1, 2, 3, 4];
      const y = [2, 3, 4, 5];
      const expected = 1 * 2 + 2 * 3 + 3 * 4 + 4 * 5; // = 2 + 6 + 12 + 20 = 40

      const result = ddot(n, x, 1, y, 1);

      expect(result).toBe(expected);
    });

    it("should handle n <= 0 (early return)", () => {
      const n = 0;
      const x = [1, 2, 3];
      const y = [4, 5, 6];

      const result = ddot(n, x, 1, y, 1);

      expect(result).toBe(0.0);
    });

    it("should handle different increments", () => {
      const n = 2;
      const x = [1, 0, 3, 0]; // effective x = [1, 3] with incx = 2
      const y = [2, 0, 4, 0]; // effective y = [2, 4] with incy = 2
      const expected = 1 * 2 + 3 * 4; // = 2 + 12 = 14

      const result = ddot(n, x, 2, y, 2);

      expect(result).toBe(expected);
    });

    it("should handle negative increments", () => {
      const n = 3;
      const x = [1, 2, 3];
      const y = [4, 5, 6];

      const result = ddot(n, x, -1, y, -1);

      // With negative increments, access is: x[2]*y[2] + x[1]*y[1] + x[0]*y[0]
      // = 3*6 + 2*5 + 1*4 = 18 + 10 + 4 = 32
      expect(result).toBe(32);
    });

    it("should handle unrolled loop optimization (n >= 5)", () => {
      const n = 10;
      const x = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
      const y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const expected = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10; // = 55

      const result = ddot(n, x, 1, y, 1);

      expect(result).toBe(expected);
    });

    it("should handle floating point precision", () => {
      const n = 3;
      const x = [0.1, 0.2, 0.3];
      const y = [0.4, 0.5, 0.6];
      const expected = 0.1 * 0.4 + 0.2 * 0.5 + 0.3 * 0.6; // = 0.04 + 0.1 + 0.18 = 0.32

      const result = ddot(n, x, 1, y, 1);

      expect(result).toBeCloseTo(expected, 10);
    });
  });

  describe("DCOPY - Vector Copy", () => {
    it("should copy vector correctly (basic case)", () => {
      const n = 4;
      const x = [1, 2, 3, 4];
      const y = [0, 0, 0, 0];
      const expected = [1, 2, 3, 4];

      dcopy(n, x, 1, y, 1);

      expect(y).toEqual(expected);
    });

    it("should handle n <= 0 (early return)", () => {
      const n = 0;
      const x = [1, 2, 3];
      const y = [10, 20, 30];
      const expected = [10, 20, 30]; // y should remain unchanged

      dcopy(n, x, 1, y, 1);

      expect(y).toEqual(expected);
    });

    it("should handle different increments", () => {
      const n = 2;
      const x = [1, 0, 3, 0]; // effective x = [1, 3] with incx = 2
      const y = [0, 99, 0, 99]; // effective y = [y[0], y[2]] with incy = 2
      const expected = [1, 99, 3, 99]; // Copy x[0]->y[0], x[2]->y[2]

      dcopy(n, x, 2, y, 2);

      expect(y).toEqual(expected);
    });

    it("should handle unrolled loop optimization (n >= 7)", () => {
      const n = 10;
      const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
      const expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

      dcopy(n, x, 1, y, 1);

      expect(y).toEqual(expected);
    });
  });

  describe("DSCAL - Vector Scale", () => {
    it("should scale vector correctly (basic case)", () => {
      const n = 4;
      const alpha = 2.0;
      const x = [1, 2, 3, 4];
      const expected = [2, 4, 6, 8];

      dscal(n, alpha, x, 1);

      expect(x).toEqual(expected);
    });

    it("should handle alpha = 1 (early return)", () => {
      const n = 3;
      const alpha = 1.0;
      const x = [1, 2, 3];
      const expected = [1, 2, 3]; // x should remain unchanged

      dscal(n, alpha, x, 1);

      expect(x).toEqual(expected);
    });

    it("should handle n <= 0 (early return)", () => {
      const n = 0;
      const alpha = 2.0;
      const x = [1, 2, 3];
      const expected = [1, 2, 3]; // x should remain unchanged

      dscal(n, alpha, x, 1);

      expect(x).toEqual(expected);
    });

    it("should handle incx <= 0 (early return)", () => {
      const n = 3;
      const alpha = 2.0;
      const x = [1, 2, 3];
      const expected = [1, 2, 3]; // x should remain unchanged

      dscal(n, alpha, x, 0);

      expect(x).toEqual(expected);
    });

    it("should handle different increment", () => {
      const n = 2;
      const alpha = 3.0;
      const x = [1, 99, 2, 99]; // effective x = [1, 2] with incx = 2
      const expected = [3, 99, 6, 99]; // Scale x[0] and x[2]

      dscal(n, alpha, x, 2);

      expect(x).toEqual(expected);
    });

    it("should handle unrolled loop optimization (n >= 5)", () => {
      const n = 8;
      const alpha = 0.5;
      const x = [2, 4, 6, 8, 10, 12, 14, 16];
      const expected = [1, 2, 3, 4, 5, 6, 7, 8];

      dscal(n, alpha, x, 1);

      expect(x).toEqual(expected);
    });
  });

  describe("DSWAP - Vector Swap", () => {
    it("should swap vectors correctly (basic case)", () => {
      const n = 4;
      const x = [1, 2, 3, 4];
      const y = [10, 20, 30, 40];
      const expectedX = [10, 20, 30, 40];
      const expectedY = [1, 2, 3, 4];

      dswap(n, x, 1, y, 1);

      expect(x).toEqual(expectedX);
      expect(y).toEqual(expectedY);
    });

    it("should handle n <= 0 (early return)", () => {
      const n = 0;
      const x = [1, 2, 3];
      const y = [10, 20, 30];
      const expectedX = [1, 2, 3];
      const expectedY = [10, 20, 30];

      dswap(n, x, 1, y, 1);

      expect(x).toEqual(expectedX);
      expect(y).toEqual(expectedY);
    });

    it("should handle different increments", () => {
      const n = 2;
      const x = [1, 99, 3, 99]; // effective x = [1, 3] with incx = 2
      const y = [10, 88, 30, 88]; // effective y = [10, 30] with incy = 2
      const expectedX = [10, 99, 30, 99];
      const expectedY = [1, 88, 3, 88];

      dswap(n, x, 2, y, 2);

      expect(x).toEqual(expectedX);
      expect(y).toEqual(expectedY);
    });

    it("should handle unrolled loop optimization (n >= 3)", () => {
      const n = 6;
      const x = [1, 2, 3, 4, 5, 6];
      const y = [10, 20, 30, 40, 50, 60];
      const expectedX = [10, 20, 30, 40, 50, 60];
      const expectedY = [1, 2, 3, 4, 5, 6];

      dswap(n, x, 1, y, 1);

      expect(x).toEqual(expectedX);
      expect(y).toEqual(expectedY);
    });
  });

  describe("DNRM2 - Euclidean Norm", () => {
    it("should compute norm correctly (basic case)", () => {
      const n = 3;
      const x = [3, 4, 0]; // ||x||_2 = sqrt(3^2 + 4^2 + 0^2) = 5
      const expected = 5;

      const result = dnrm2(n, x, 1);

      expect(result).toBeCloseTo(expected, 10);
    });

    it("should handle n <= 0 (early return)", () => {
      const n = 0;
      const x = [1, 2, 3];

      const result = dnrm2(n, x, 1);

      expect(result).toBe(0.0);
    });

    it("should handle different increments", () => {
      const n = 2;
      const x = [3, 0, 4, 0]; // effective x = [3, 4] with incx = 2
      const expected = 5; // sqrt(3^2 + 4^2) = 5

      const result = dnrm2(n, x, 2);

      expect(result).toBeCloseTo(expected, 10);
    });

    it("should handle all zeros", () => {
      const n = 3;
      const x = [0, 0, 0];

      const result = dnrm2(n, x, 1);

      expect(result).toBe(0.0);
    });

    it("should handle numerical stability with scaling", () => {
      const n = 2;
      const x = [1e10, 1e10]; // Large values that could cause overflow
      const expected = Math.sqrt(2) * 1e10;

      const result = dnrm2(n, x, 1);

      expect(result).toBeCloseTo(expected, 5); // Less precision due to scaling
    });
  });

  describe("DASUM - Sum of Absolute Values", () => {
    it("should compute sum of absolute values correctly (basic case)", () => {
      const n = 4;
      const x = [1, -2, 3, -4];
      const expected = 1 + 2 + 3 + 4; // = 10

      const result = dasum(n, x, 1);

      expect(result).toBe(expected);
    });

    it("should handle n <= 0 (early return)", () => {
      const n = 0;
      const x = [1, 2, 3];

      const result = dasum(n, x, 1);

      expect(result).toBe(0.0);
    });

    it("should handle incx <= 0 (early return)", () => {
      const n = 3;
      const x = [1, 2, 3];

      const result = dasum(n, x, 0);

      expect(result).toBe(0.0);
    });

    it("should handle different increments", () => {
      const n = 2;
      const x = [-1, 99, -3, 99]; // effective x = [-1, -3] with incx = 2
      const expected = 1 + 3; // = 4

      const result = dasum(n, x, 2);

      expect(result).toBe(expected);
    });

    it("should handle unrolled loop optimization (n >= 6)", () => {
      const n = 8;
      const x = [-1, -2, -3, -4, -5, -6, -7, -8];
      const expected = 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8; // = 36

      const result = dasum(n, x, 1);

      expect(result).toBe(expected);
    });
  });

  describe("IDAMAX - Index of Maximum Absolute Value", () => {
    it("should find index of maximum absolute value (basic case)", () => {
      const n = 4;
      const x = [1, -5, 3, 2]; // max |x[i]| = |-5| = 5 at index 1
      const expected = 1; // 0-based index

      const result = idamax(n, x, 1);

      expect(result).toBe(expected);
    });

    it("should handle n < 1 (early return)", () => {
      const n = 0;
      const x = [1, 2, 3];

      const result = idamax(n, x, 1);

      expect(result).toBe(0);
    });

    it("should handle incx <= 0 (early return)", () => {
      const n = 3;
      const x = [1, 2, 3];

      const result = idamax(n, x, 0);

      expect(result).toBe(0);
    });

    it("should handle n == 1", () => {
      const n = 1;
      const x = [42];

      const result = idamax(n, x, 1);

      expect(result).toBe(0);
    });

    it("should handle different increments", () => {
      const n = 3;
      const x = [1, 99, -10, 99, 2, 99]; // effective x = [1, -10, 2] with incx = 2
      const expected = 1; // max |x[i]| = |-10| = 10 at index 1

      const result = idamax(n, x, 2);

      expect(result).toBe(expected);
    });

    it("should return first occurrence for ties", () => {
      const n = 3;
      const x = [5, -5, 3]; // both |5| and |-5| are maximum, should return first
      const expected = 0;

      const result = idamax(n, x, 1);

      expect(result).toBe(expected);
    });
  });

  describe("DROTG - Generate Plane Rotation", () => {
    it("should generate rotation to zero out b (basic case)", () => {
      const a = 3;
      const b = 4;

      const { c, s, r } = drotg(a, b);

      // Check that rotation zeros out b: c*a + s*b should equal r
      // and -s*a + c*b should be 0 (or very close to 0)
      expect(c * a + s * b).toBeCloseTo(r, 10);
      expect(-s * a + c * b).toBeCloseTo(0, 10);

      // Check that c^2 + s^2 = 1
      expect(c * c + s * s).toBeCloseTo(1, 10);

      // For this case, r should be 5 (sqrt(3^2 + 4^2))
      expect(Math.abs(r)).toBeCloseTo(5, 10);
    });

    it("should handle b = 0", () => {
      const a = 5;
      const b = 0;

      const { c, s, r, z } = drotg(a, b);

      expect(c).toBe(1.0);
      expect(s).toBe(0.0);
      expect(r).toBe(a);
      expect(z).toBe(0.0);
    });

    it("should handle a = 0", () => {
      const a = 0;
      const b = 3;

      const { c, s, r, z } = drotg(a, b);

      expect(c).toBe(0.0);
      expect(s).toBe(1.0);
      expect(r).toBe(b);
      expect(z).toBe(1.0);
    });

    it("should handle both a and b non-zero", () => {
      const a = 1;
      const b = 1;

      const { c, s, r } = drotg(a, b);

      // c^2 + s^2 should equal 1
      expect(c * c + s * s).toBeCloseTo(1, 10);

      // r should be sqrt(1^2 + 1^2) = sqrt(2)
      expect(Math.abs(r)).toBeCloseTo(Math.sqrt(2), 10);
    });
  });

  describe("DROT - Apply Plane Rotation", () => {
    it("should apply rotation correctly (basic case)", () => {
      const n = 2;
      const x = [1, 2];
      const y = [3, 4];
      const c = Math.cos(Math.PI / 4); // 45 degrees
      const s = Math.sin(Math.PI / 4);

      // Store original values
      const x0 = [...x];
      const y0 = [...y];

      drot(n, x, 1, y, 1, c, s);

      // Check rotation: [x'] = [c  s] [x]
      //                 [y']   [-s c] [y]
      expect(x[0]).toBeCloseTo(c * x0[0] + s * y0[0], 10);
      expect(y[0]).toBeCloseTo(c * y0[0] - s * x0[0], 10);
      expect(x[1]).toBeCloseTo(c * x0[1] + s * y0[1], 10);
      expect(y[1]).toBeCloseTo(c * y0[1] - s * x0[1], 10);
    });

    it("should handle n <= 0 (early return)", () => {
      const n = 0;
      const x = [1, 2, 3];
      const y = [4, 5, 6];
      const expectedX = [1, 2, 3];
      const expectedY = [4, 5, 6];

      drot(n, x, 1, y, 1, 0.5, 0.5);

      expect(x).toEqual(expectedX);
      expect(y).toEqual(expectedY);
    });

    it("should handle different increments", () => {
      const n = 2;
      const x = [1, 99, 2, 99]; // effective x = [1, 2] with incx = 2
      const y = [3, 88, 4, 88]; // effective y = [3, 4] with incy = 2
      const c = 1.0;
      const s = 0.0; // Identity rotation

      drot(n, x, 2, y, 2, c, s);

      // With identity rotation, x and y should remain unchanged
      expect(x).toEqual([1, 99, 2, 99]);
      expect(y).toEqual([3, 88, 4, 88]);
    });

    it("should handle identity rotation (c=1, s=0)", () => {
      const n = 3;
      const x = [1, 2, 3];
      const y = [4, 5, 6];
      const expectedX = [1, 2, 3];
      const expectedY = [4, 5, 6];

      drot(n, x, 1, y, 1, 1.0, 0.0);

      expect(x).toEqual(expectedX);
      expect(y).toEqual(expectedY);
    });

    it("should handle 90-degree rotation (c=0, s=1)", () => {
      const n = 2;
      const x = [1, 2];
      const y = [3, 4];

      drot(n, x, 1, y, 1, 0.0, 1.0);

      // 90-degree rotation: x' = y, y' = -x
      expect(x).toEqual([3, 4]);
      expect(y).toEqual([-1, -2]);
    });
  });
});
