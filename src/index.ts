/**
 * BLAS-TS: Pure TypeScript implementation of BLAS (Basic Linear Algebra Subprograms)
 *
 * This package provides high-performance linear algebra operations implemented
 * in pure TypeScript, following the reference FORTRAN BLAS implementations.
 */

// Export types
export { Vector } from "./types";

// Export Level 1 BLAS operations
export {
  daxpy,
  drotg,
  drot,
  dscal,
  dasum,
  dcopy,
  dswap,
  ddot,
  dnrm2,
  idamax,
} from "./level1";

// Export Level 2 BLAS operations
export { dgemv, dger, dsyr, dsyr2, dtrmv, dtrsv, dsymv } from "./level2";

// Export Level 3 BLAS operations
export { dgemm, dsymm, dtrmm, dtrsm, dsyrk, dsyr2k } from "./level3";

// Re-export for convenience - users can import { daxpy } from 'blas-ts'
//export { daxpy as default } from './level1';
