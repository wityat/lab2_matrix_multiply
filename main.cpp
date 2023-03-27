#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <mpi.h>

class Matrix {
public:
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), data_(rows * cols) {}

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    double* data() {
        return data_.data();
    }

    const double* data() const {
        return data_.data();
    }

    double& operator()(size_t row, size_t col) { return data_[row * cols_ + col]; }
    const double& operator()(size_t row, size_t col) const { return data_[row * cols_ + col]; }

    void randomize(double lower_bound = -1, double upper_bound = 1) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(lower_bound, upper_bound);

        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                (*this)(i, j) = dis(gen);
            }
        }
    }

    void print() const {
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) {
                std::cout << std::setw(10) << std::setprecision(4) << (*this)(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
    Matrix slice(size_t row_start, size_t col_start, size_t row_end, size_t col_end) const {
        if (row_start >= row_end || col_start >= col_end || row_end > rows() || col_end > cols()) {
            throw std::out_of_range("Invalid slice range");
        }

        size_t new_rows = row_end - row_start;
        size_t new_cols = col_end - col_start;
        Matrix new_matrix(new_rows, new_cols);

        for (size_t i = 0; i < new_rows; ++i) {
            for (size_t j = 0; j < new_cols; ++j) {
                new_matrix(i, j) = (*this)(row_start + i, col_start + j);
            }
        }

        return new_matrix;
    }

    size_t size() const {
        return cols_ * rows_;
    }


private:
    size_t rows_;
    size_t cols_;
    std::vector<double> data_;
};

Matrix matrix_multiply_mpi_row(const Matrix& A, const Matrix& B, int rank, int size) {
    size_t rows = A.rows();
    size_t rows_per_process = rows / size;
    size_t remaining_rows = rows % size;

    // Определить количество строк для этого процесса
    size_t local_rows = rows_per_process;
    if (rank < remaining_rows) {
        local_rows++;
    }

    Matrix local_A(local_rows, A.cols());
    Matrix local_C(local_rows, B.cols());

    // Подготовить количество и смещения для операций MPI_Scatterv и MPI_Gatherv
    std::vector<int> send_counts(size, rows_per_process * A.cols());
    std::vector<int> displs(size, 0);
    for (int i = 0; i < size; ++i) {
        if (i < remaining_rows) {
            send_counts[i] += A.cols();
        }
        if (i > 0) {
            displs[i] = displs[i - 1] + send_counts[i - 1];
        }
    }

    // Разослать строки матрицы A по всем процессам
    MPI_Scatterv(A.data(), send_counts.data(), displs.data(), MPI_DOUBLE, local_A.data(), local_rows * A.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Передать матрицу B всем процессам
    MPI_Bcast(const_cast<double*>(B.data()), B.rows() * B.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Вычислить локальный блок матрицы C
    for (size_t i = 0; i < local_rows; ++i) {
        for (size_t j = 0; j < B.cols(); ++j) {
            double sum = 0;
            for (size_t k = 0; k < A.cols(); ++k) {
                sum += local_A(i, k) * B(k, j);
            }
            local_C(i, j) = sum;
        }
    }

    // Собрать все локальные блоки матрицы C на главном процессе
    Matrix C(A.rows(), B.cols());
    std::vector<int> recv_counts = send_counts;
    for (int i = 0; i < size; ++i) {
        recv_counts[i] = recv_counts[i] / A.cols() * B.cols();
    }
    std::vector<int> recv_displs(size, 0);
    for (int i = 1; i < size; ++i) {
        recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
    }

    MPI_Gatherv(local_C.data(), local_rows * B.cols(), MPI_DOUBLE, C.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return C;
}

Matrix matrix_multiply_mpi_column(const Matrix& A, const Matrix& B, int rank, int size) {
    // Вычислить количество столбцов матрицы B, которые должны быть обработаны каждым процессом
    size_t cols_per_process = B.cols() / size;
    size_t remaining_cols = B.cols() % size;
    size_t local_cols = cols_per_process + (rank < remaining_cols ? 1 : 0);

    // Создать локальные матрицы для хранения частей матрицы B и результата умножения
    Matrix local_B(A.cols(), local_cols);
    Matrix local_C(A.rows(), local_cols);

    if (rank == 0) {
        // Мастер-процесс отправляет части матрицы B другим процессам
        size_t current_col = cols_per_process + (remaining_cols > 0 ? 1 : 0);
        for (int i = 1; i < size; ++i) {
            size_t start_col = current_col;
            size_t end_col = start_col + cols_per_process + (i < remaining_cols ? 1 : 0);
            Matrix segment = B.slice(0, start_col, B.rows(), end_col);
            MPI_Send(segment.data(), segment.rows() * segment.cols(), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            current_col = end_col;
        }
        // Мастер-процесс сохраняет свою часть матрицы B
        local_B = B.slice(0, 0, B.rows(), local_cols);
    } else {
        // Рабочие процессы получают свои части матрицы B от мастер-процесса
        MPI_Recv(local_B.data(), local_B.rows() * local_B.cols(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Все процессы получают матрицу A от мастер-процесса
    MPI_Bcast(const_cast<double *>(A.data()), A.rows() * A.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Каждый процесс вычисляет свою часть результата умножения матриц A и B
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < local_cols; ++j) {
            double sum = 0;
            for (size_t k = 0; k < A.cols(); ++k) {
                sum += A(i, k) * local_B(k, j);
            }
            local_C(i, j) = sum;
        }
    }

    // Создать матрицу C для хранения результата умножения
    Matrix C(A.rows(), B.cols());
    if (rank == 0) {
        // Мастер-процесс сохраняет свою часть результата в матрице C
        for (size_t j = 0; j < local_cols; ++j) {
            for (size_t i = 0; i < A.rows(); ++i) {
                C(i, j) = local_C(i, j);
            }
        }
        // Мастер-процесс получает части результата от других процессов и сохраняет их в матрице C
        size_t current_col = local_cols;
        for (int i = 1; i < size; ++i) {
            size_t start_col = current_col;
            size_t end_col = start_col + cols_per_process + (i < remaining_cols ? 1 : 0);
            size_t recv_cols = end_col - start_col;
            Matrix recv_matrix(A.rows(), recv_cols);
            MPI_Recv(recv_matrix.data(), recv_matrix.rows() * recv_matrix.cols(), MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            for (size_t j = 0; j < recv_cols; ++j) {
                for (size_t i = 0; i < A.rows(); ++i) {
                    C(i, start_col + j) = recv_matrix(i, j);
                }
            }
            current_col = end_col;
        }
    } else {
        // Рабочие процессы отправляют свои части результата мастер-процессу
        MPI_Send(local_C.data(), local_C.rows() * local_C.cols(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    return C;
}

Matrix matrix_multiply_mpi_block(const Matrix& A, const Matrix& B, int rank, int size) {
    const int N = A.rows();
    const int block_size = N / size;
    const int remainder = N % size;

    // Буфер для локального блока матрицы A.
    std::vector<double> A_local((block_size + (rank < remainder ? 1 : 0)) * N);

    // Рассчитать смещения и количества элементов для операций scatterv и gatherv.
    std::vector<int> displacements(size, 0);
    std::vector<int> counts(size, 0);

    for (int i = 0; i < size; ++i) {
        displacements[i] = i * block_size * N + std::min(i, remainder) * N;
        counts[i] = block_size * N + (i < remainder ? N : 0);
    }

    // Разослать матрицу A по всем процессам.
    MPI_Scatterv(A.data(), counts.data(), displacements.data(), MPI_DOUBLE, A_local.data(), A_local.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Передать матрицу B всем процессам.
    MPI_Bcast(const_cast<double *>(B.data()), B.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Вычислить локальный блок матрицы C.
    Matrix C_local = Matrix(block_size + (rank < remainder ? 1 : 0), B.cols());
    for (int i = 0; i < C_local.rows(); ++i) {
        for (int j = 0; j < B.cols(); ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A_local[i * N + k] * B(k, j);
            }
            C_local(i, j) = sum;
        }
    }

    // Собрать все локальные блоки матрицы C на главном процессе.
    Matrix C = Matrix(N, B.cols());

    // Рассчитать смещения и количества элементов для операции gatherv.
    for (int i = 0; i < size; ++i) {
        displacements[i] = i * block_size * B.cols() + std::min(i, remainder) * B.cols();
        counts[i] = block_size * B.cols() + (i < remainder ? B.cols() : 0);
    }

    MPI_Gatherv(C_local.data(), C_local.size(), MPI_DOUBLE, C.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return C;
}

std::string check_multiplication(const Matrix& A, const Matrix& B, const Matrix& C) {
    const double EPSILON = 1e-6;

    if (A.cols() != B.rows() || A.rows() != C.rows() || B.cols() != C.cols()) {
        return "incorrect";
    }

    Matrix ref_C(A.rows(), B.cols());
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < B.cols(); ++j) {
            double sum = 0;
            for (size_t k = 0; k < A.cols(); ++k) {
                sum += A(i, k) * B(k, j);
            }
            ref_C(i, j) = sum;
        }
    }

    for (size_t i = 0; i < C.rows(); ++i) {
        for (size_t j = 0; j < C.cols(); ++j) {
            if (std::abs(C(i, j) - ref_C(i, j)) > EPSILON) {
                return "incorrect";
            }
        }
    }

    return "correct";
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " matrix_size " << "algo_name" << std::endl;
        return 1;
    }

    size_t matrix_size = std::stoi(argv[1]);
    std:: string algo = std::string(argv[2]);

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Matrix A(matrix_size, matrix_size);
    Matrix B(matrix_size, matrix_size);
    Matrix C(matrix_size, matrix_size);

    A.randomize();
    B.randomize();

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::steady_clock::now();

    if (algo == "row") {
        C = matrix_multiply_mpi_row(A, B, rank, size);
    } else if (algo == "column") {
        C = matrix_multiply_mpi_column(A, B, rank, size);
    } else if (algo == "block") {
        C = matrix_multiply_mpi_block(A, B, rank, size);
    }
    else if (rank == 0){
        std::cerr << "Algo name could be either row, column or block." << std::endl;
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_seconds).count();
    if (rank == 0) {
        std::cout << check_multiplication(A, B, C) << " " << elapsed_ms << std::endl;
    }

    MPI_Finalize();

    return 0;
}
