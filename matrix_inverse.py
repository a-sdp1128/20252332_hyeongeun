
import math


# 행렬식 계산 함수
def determinant(matrix):
    n = len(matrix)

    if n == 0:
        return 1

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return (
            matrix[0][0] * matrix[1][1]
            - matrix[0][1] * matrix[1][0]
        )

    det = 0
    for j in range(n):
        minor = [
            row[:j] + row[j + 1:]
            for row in matrix[1:]
        ]
        det += ((-1) ** j) * matrix[0][j] * determinant(minor)

    return det


# 소행렬 생성 함수
def get_minor(matrix, row, col):
    return [
        line[:col] + line[col + 1:]
        for i, line in enumerate(matrix)
        if i != row
    ]


# 행렬식 역행렬 계산
def inverse_by_determinant(matrix):
    n = len(matrix)
    det = determinant(matrix)

    if math.isclose(det, 0.0, abs_tol=1e-10):
        raise ValueError("역행렬이 존재하지 않습니다.")

    cofactors = []

    for i in range(n):
        row = []
        for j in range(n):
            minor = get_minor(matrix, i, j)
            cofactor = ((-1) ** (i + j)) * determinant(minor)
            row.append(cofactor)
        cofactors.append(row)

    # 수반행렬
    adjugate = [
        [cofactors[j][i] for j in range(n)]
        for i in range(n)
    ]

    return [
        [adjugate[i][j] / det for j in range(n)]
        for i in range(n)
    ]


# 가우스-조던 소거법
def inverse_by_gauss_jordan(matrix):
    n = len(matrix)

    # 단위행렬 결합
    a = [
        [float(value) for value in row]
        for row in matrix
    ]
    identity = [
        [float(i == j) for j in range(n)]
        for i in range(n)
    ]
    augmented = [
        a[i] + identity[i]
        for i in range(n)
    ]

    for col in range(n):
        # 부분 피벗팅
        pivot_row = max(
            range(col, n),
            key=lambda r: abs(augmented[r][col])
        )

        if math.isclose(
            augmented[pivot_row][col],
            0.0,
            abs_tol=1e-10
        ):
            raise ValueError("역행렬이 존재하지 않습니다.")

        augmented[col], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[col]
        )

        # 피벗 원소를 1로
        pivot = augmented[col][col]
        augmented[col] = [
            value / pivot for value in augmented[col]
        ]

        # 나머지 행의 해당 열 원소를 0으로
        for row in range(n):
            if row != col:
                factor = augmented[row][col]
                augmented[row] = [
                    augmented[row][j]
                    - factor * augmented[col][j]
                    for j in range(2 * n)
                ]

    return [row[n:] for row in augmented]


# 역행렬 출력 함수
def print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{value:10.4f}" for value in row))


# 역행렬 비교 함수
def compare_matrices(a, b, tol=1e-8):
    return all(
        math.isclose(a[i][j], b[i][j],
                     rel_tol=tol, abs_tol=tol)
        for i in range(len(a))
        for j in range(len(a))
    )


def main():

    try:
        n = int(input("행렬의 크기 n을 입력하세요: "))
        if n <= 0:
            print("오류: n은 양의 정수여야 합니다.")
            return

        matrix = []
        print("각 행의 원소를 공백으로 구분하여 입력하세요.")

        for i in range(n):
            while True:
                try:
                    row = list(map(
                        int,
                        input(f"{i + 1}행: ").split()
                    ))
                    if len(row) != n:
                        print(f"오류: 원소를 {n}개 입력하세요.")
                        continue
                    matrix.append(row)
                    break
                except ValueError:
                    print("오류: 정수만 입력하세요.")

        print("\n입력한 행렬:")
        print_matrix(matrix)

        print("\n[방법 1] 행렬식과 여인수 전개")
        try:
            inverse1 = inverse_by_determinant(matrix)
            print("역행렬:")
            print_matrix(inverse1)
        except ValueError as e:
            inverse1 = None
            print(e)

        print("\n[방법 2] 가우스-조던 소거법")
        try:
            inverse2 = inverse_by_gauss_jordan(matrix)
            print("역행렬:")
            print_matrix(inverse2)
        except ValueError as e:
            inverse2 = None
            print(e)

        print("\n[결과 비교]")
        if inverse1 is not None and inverse2 is not None:
            if compare_matrices(inverse1, inverse2):
                print("두 방법의 역행렬이 동일합니다.")
            else:
                print("두 방법의 결과가 다릅니다.")
        else:
            print("역행렬이 존재하지 않아 비교할 수 없습니다.")

    except ValueError:
        print("오류: n에는 정수를 입력해야 합니다.")


if __name__ == "__main__":
    main()
