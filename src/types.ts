export type Vector = number[];
export type Matrix = number[]; // Column-major storage: A[i,j] = A[i + j * ldA]
// BLAS-specific types following Fortran conventions
export enum BLASTranspose {
  NoTranspose,
  Transpose,
  ConjugateTranspose,
}
export enum BLASUplo {
  Upper,
  Lower,
}
export enum BLASDiag {
  NonUnit,
  Unit,
}
export enum BLASSide {
  Left,
  Right,
}

// Complex number representation
export interface Complex {
  real: number;
  imag: number;
}
