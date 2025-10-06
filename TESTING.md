# Testing Setup

## Overview

This project uses **Jest** as the testing framework with TypeScript support via `ts-jest`.

## Available Test Scripts

```bash
# Run all tests
pnpm test

# Run tests in watch mode (re-runs when files change)
pnpm test:watch

# Run tests with coverage report
pnpm test:coverage

# Run tests in CI mode (no watch, with coverage)
pnpm test:ci
```

## Test Structure

- Tests are located in the `tests/` directory
- Test files should follow the naming convention: `*.test.ts` or `*.spec.ts`
- The current test file `tests/level1.test.ts` contains comprehensive tests for the `daxpy` function

## Sample Test for `daxpy`

The `daxpy` function test includes:

- Basic functionality test (`y = alpha*x + y`)
- Edge cases: `alpha = 0`, `alpha = 1`, negative alpha
- Empty vectors and single element vectors

## Coverage

Coverage reports are generated in the `coverage/` directory when running `pnpm test:coverage`. The reports include:

- Text output in terminal
- HTML report (open `coverage/lcov-report/index.html`)
- LCOV format for CI integration

## Configuration Files

- `jest.config.js` - Main Jest configuration
- `tsconfig.test.json` - TypeScript configuration for tests (extends main tsconfig)

## Adding New Tests

Create new test files in the `tests/` directory following this pattern:

```typescript
import { functionName } from "../src/module";

describe("Module Name", () => {
  describe("functionName", () => {
    it("should do something", () => {
      // Arrange
      const input = "test";
      const expected = "expected result";

      // Act
      const result = functionName(input);

      // Assert
      expect(result).toBe(expected);
    });
  });
});
```
