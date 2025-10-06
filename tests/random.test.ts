import { Random } from "random-js";
import { dgemm, dsyrk } from "../src/level3";
import { BLASTranspose, BLASUplo } from "../src/types";

const mimGenerator =
  (ld: number) =>
  (i: number, j: number): number => {
    return i + j * ld;
  };

const random = new Random();
describe("Random", () => {
  describe("DGEMM (Double precision general matrix multiply)", () => {
    it("should multiply two diagonal 2x2 matrices", () => {
      const d1 = random.real(0, 1);
      const d2 = random.real(0, 1);
      const d3 = random.real(0, 1);
      const d4 = random.real(0, 1);
      let A = [d1, 0, 0, d2];
      let B = [d3, 0, 0, d4];
      let C = [0.0, 0.0, 0.0, 0.0];
      let expected = [d1 * d3, 0, 0, d2 * d4];
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
      expect(C).toEqual(expected);
    });

    it("should multiply two 2x2 matrices", () => {
      let A = new Array(4).fill(0).map(() => random.real(0, 1));
      let B = new Array(4).fill(0).map(() => random.real(0, 1));
      let C = [0.0, 0.0, 0.0, 0.0];
      const mim = mimGenerator(2);
      let expected = [
        A[mim(0, 0)] * B[mim(0, 0)] + A[mim(0, 1)] * B[mim(1, 0)],
        A[mim(1, 0)] * B[mim(0, 0)] + A[mim(1, 1)] * B[mim(1, 0)],
        A[mim(0, 0)] * B[mim(0, 1)] + A[mim(0, 1)] * B[mim(1, 1)],
        A[mim(1, 0)] * B[mim(0, 1)] + A[mim(1, 1)] * B[mim(1, 1)],
      ];
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
      expect(C).toEqual(expected);
    });
  });

  describe("DGEMM (Double precision general matrix multiply) with transpose", () => {
    it("should multiply two 3x3 matrices", () => {
      const A = [1, 2, 4, 2, 3, 5, 4, 5, 6];
      const B = [1, 2, 4, 2, 3, 5, 4, 5, 6];
      const C = [0, 0, 0, 0, 0, 0, 0, 0, 0];
      const expected = [21, 28, 38, 28, 38, 53, 38, 53, 77];
      dgemm(
        BLASTranspose.NoTranspose,
        BLASTranspose.NoTranspose,
        3,
        3,
        3,
        1.0,
        A,
        3,
        B,
        3,
        0.0,
        C,
        3
      );

      console.log(C);

      expect(C).toEqual(expected);
    });
  });

  describe("DSYRK (Double precision symmetric rank-k update)", () => {
    it("should multiply two 3x3 matrices", () => {
      const A = [1, 2, 4, 2, 3, 5, 4, 5, 6];
      const C = [0, 0, 0, 0, 0, 0, 0, 0, 0];
      const expected = [21, 0, 0, 28, 38, 0, 38, 53, 77];

      dsyrk(
        BLASUplo.Upper,
        BLASTranspose.NoTranspose,
        3,
        3,
        1,
        A,
        3,
        100,
        C,
        3
      );
      console.log(C);
      expect(C).toEqual(expected);
    });
    it("should compute C = alpha*A*A^T + beta*C with upper triangle", () => {
      const A = [1, 2, 4, 2, 3, 5, 4, 5, 6];
      const C1 = [1, 1, 1, 1, 1, 1, 1, 1, 1];
      const C2 = [1, 0, 0, 1, 1, 0, 1, 1, 1];

      dsyrk(BLASUplo.Upper, BLASTranspose.NoTranspose, 3, 3, 1, A, 3, 1, C1, 3);
      dsyrk(BLASUplo.Upper, BLASTranspose.NoTranspose, 3, 3, 1, A, 3, 1, C2, 3);

      expect(C1[0]).toEqual(C2[0]);
      expect(C1[3]).toEqual(C2[3]);
      expect(C1[4]).toEqual(C2[4]);
      expect(C1[6]).toEqual(C2[6]);
      expect(C1[7]).toEqual(C2[7]);
      expect(C1[8]).toEqual(C2[8]);
    });
  });
});
