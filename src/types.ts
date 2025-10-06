// BLAS-specific types following Fortran conventions
export enum Transpose {
  NoTranspose = "N",
  Transpose = "T",
  ConjugateTranspose = "C",
}
export enum Triangular {
  Upper = "U",
  Lower = "L",
}
export enum Diagonal {
  NonUnit = "N",
  Unit = "U",
}
export enum Side {
  Left = "L",
  Right = "R",
}

// Complex number representation
export interface Complex {
  real: number;
  imag: number;
}
