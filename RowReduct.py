import math

import numpy as np


class RowReduction:
    def __init__(self, matrix):
        self.matrix = matrix
        self.rowCount = np.size(matrix)
        if self.rowCount == 0:
            self.columnCount = 0
        else:
            self.columnCount = len(matrix[0])

    def scale(self, row, scalor):
        self.matrix[row] = self.matrix[row] * scalor

    def swaprow(self, row1, row2):
        self.matrix[[row1, row2]] = self.matrix[[row2, row1]]

    def replacerow(self, replacement_row, scalor_row, scalor):
        self.matrix[replacement_row] -= self.matrix[scalor_row] * scalor

    def is_reduced_row_echelon_form(self):
        rows, cols = self.matrix.shape
        leading_one_col = -1

        for i in range(rows):
            row = self.matrix[i]
            nonzero_indices = np.where(row != 0)[0]

            if len(nonzero_indices) == 0:
                continue

            leading_one = nonzero_indices[0]
            if row[leading_one] != 1:
                return False

            for j in range(rows):
                if j != i and self.matrix[j, leading_one] != 0:
                    return False

            if leading_one <= leading_one_col:
                return False

            leading_one_col = leading_one

        return True

    def to_reduced_echelon_form(self):
        rows, cols = self.matrix.shape
        lead = 0
        for r in range(rows):
            if lead >= cols:
                return
            i = r
            while self.matrix[i, lead] == 0:
                i += 1
                if i == rows:
                    i = r
                    lead += 1
                    if lead == cols:
                        return
            self.swaprow(r, i)
            self.scale(r, 1 / self.matrix[r, lead])

            for i in range(rows):
                if i != r:
                    self.replacerow(i, r, self.matrix[i, lead])

            lead += 1


# Example Usage
matrixA = np.array([[10**(-16), -1, 1, 3, 4], [1, 0, 2, -4, -5], [100 ** 16, 2, 1, -15, 16]])
RE = RowReduction(matrixA)
print(RE.matrix)
RE.to_reduced_echelon_form()
print(RE.matrix)
print("Is RREF:", RE.is_reduced_row_echelon_form())
