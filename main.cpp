#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <mpi.h>
#include <random>

using namespace std;
using namespace std::chrono;

void init_matrix(vector<vector<int>> &mat, int rows, int cols);
void matrix_mult_row_split(const vector<vector<int>> &A, const vector<vector<int>> &B, vector<vector<int>> &C, int rank, int size);
void matrix_mult_col_split(const vector<vector<int>> &A, const vector<vector<int>> &B, vector<vector<int>> &C, int rank, int size);
void matrix_mult_block_split(const vector<vector<int>> &A, const vector<vector<int>> &B, vector<vector<int>> &C, int rank, int size);
bool check_matrix_mult(const vector<vector<int>> &A, const vector<vector<int>> &B, const vector<vector<int>> &C);

//int main(int argc, char *argv[]) {
//    MPI_Init(&argc, &argv);
//
//    int rank, size;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &size);
//
//    vector<int> matrix_sizes = {500, 1000, 2000}; // Размеры матриц для сравнения
//
//    if (rank == 0) {
//        cout << setw(10) << "Size" << setw(15) << "Row Split" << setw(15) << "Col Split" << setw(15) << "Block Split" << endl;
//    }
//
//    for (int n : matrix_sizes) {
//        vector<vector<int>> A(n, vector<int>(n));
//        vector<vector<int>> B(n, vector<int>(n));
//        vector<vector<int>> C(n, vector<int>(n));
//
//        init_matrix(A, n, n);
//        init_matrix(B, n, n);
//
//        high_resolution_clock::time_point t1, t2;
//
//        t1 = high_resolution_clock::now();
//        matrix_mult_row_split(A, B, C, rank, size);
//        t2 = high_resolution_clock::now();
//        auto row_duration = duration_cast<milliseconds>(t2 - t1).count();
//
//        t1 = high_resolution_clock::now();
//        matrix_mult_col_split(A, B, C, rank, size);
//        t2 = high_resolution_clock::now();
//        auto col_duration = duration_cast<milliseconds>(t2 - t1).count();
//
//        t1 = high_resolution_clock::now();
//        matrix_mult_block_split(A, B, C, rank, size);
//        t2 = high_resolution_clock::now();
//        auto block_duration = duration_cast<milliseconds>(t2 - t1).count();
//
//        if (rank == 0) {
//            cout << setw(10) << n << setw(15) << row_duration << setw(15) << col_duration << setw(15) << block_duration << endl;
//        }
//    }
//
//    MPI_Finalize();
//    return 0;
//}

//int main(int argc, char *argv[]) {
//    MPI_Init(&argc, &argv);
//
//    int rank, size;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &size);
//
//    int rows = 1000;
//    int cols = 1000;
//    vector<vector<int>> A(rows, vector<int>(cols));
//    vector<vector<int>> B(rows, vector<int>(cols));
//    vector<vector<int>> C(rows, vector<int>(cols));
//
//    init_matrix(A, rows, cols);
//    init_matrix(B, rows, cols);
//
//    high_resolution_clock::time_point t1, t2;
//
//    t1 = high_resolution_clock::now();
//    matrix_mult_row_split(A, B, C, rank, size);
//    t2 = high_resolution_clock::now();
//    auto row_duration = duration_cast<milliseconds>(t2 - t1).count();
//
//    if (rank == 0) {
//        cout << "Row split duration: " << row_duration << " ms" << endl;
//    }
//
//    t1 = high_resolution_clock::now();
//    matrix_mult_col_split(A, B, C, rank, size);
//    t2 = high_resolution_clock::now();
//    auto col_duration = duration_cast<milliseconds>(t2 - t1).count();
//
//    if (rank == 0) {
//        cout << "Column split duration: " << col_duration << " ms" << endl;
//    }
//
//    t1 = high_resolution_clock::now();
//    matrix_mult_block_split(A, B, C, rank, size);
//    t2 = high_resolution_clock::now();
//    auto block_duration = duration_cast<milliseconds>(t2 - t1).count();
//
//    if (rank == 0) {
//        cout << "Block split duration: " << block_duration << " ms" << endl;
//    }
//
//    MPI_Finalize();
//    return 0;
//}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size=4;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<int> matrix_sizes = {100, 500, 1000, 2000}; // Размеры матриц для сравнения
    vector<int> num_threads = {2, 4, 8}; // Количество потоков для сравнения
    cout << setw(10) << size ;
    if (rank == 0) {

//        cout << setw(10) << "Size" << setw(10) << "Threads" << setw(15) << "Row Split" << setw(15) << "Col Split" << setw(15) << "Block Split" << endl;
    }

    for (int n : matrix_sizes) {
        for (int t : num_threads) {
            vector<vector<int>> A(n, vector<int>(n));
            vector<vector<int>> B(n, vector<int>(n));
            vector<vector<int>> C(n, vector<int>(n));

            init_matrix(A, n, n);
            init_matrix(B, n, n);

            high_resolution_clock::time_point t1, t2;

            t1 = high_resolution_clock::now();
            matrix_mult_row_split(A, B, C, rank, t);
            t2 = high_resolution_clock::now();
            auto row_duration = duration_cast<milliseconds>(t2 - t1).count();

            t1 = high_resolution_clock::now();
//            matrix_mult_col_split(A, B, C, rank, t);
            t2 = high_resolution_clock::now();
            auto col_duration = duration_cast<milliseconds>(t2 - t1).count();

            t1 = high_resolution_clock::now();
//            matrix_mult_block_split(A, B, C, rank, t);
            t2 = high_resolution_clock::now();
            auto block_duration = duration_cast<milliseconds>(t2 - t1).count();

            if (rank == 0) {
                cout << setw(10) << n << setw(10) << t << setw(15) << row_duration << setw(15) << col_duration << setw(15) << block_duration << endl;
            }
        }
    }

    MPI_Finalize();
    return 0;
}

void init_matrix(vector<vector<int>> &mat, int rows, int cols) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 10);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i][j] = dis(gen);
        }
    }
}

void matrix_mult_row_split(const vector<vector<int>> &A, const vector<vector<int>> &B, vector<vector<int>> &C, int rank, int size) {
// Реализуйте алгоритм умножения матриц с разбиением по строкам
int rows = A.size();
int cols = B[0].size();
int k_size = B.size();

int rows_per_process = rows / size;

vector<vector<int>> local_A(rows_per_process, vector<int>(k_size));
vector<vector<int>> local_C(rows_per_process, vector<int>(cols));

MPI_Scatter(A.data()->data(), rows_per_process * k_size, MPI_INT, local_A.data()->data(), rows_per_process * k_size, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(const_cast<int*>(B.data()->data()), k_size * cols, MPI_INT, 0, MPI_COMM_WORLD);

for (int i = 0; i < rows_per_process; ++i) {
    for (int j = 0; j < cols; ++j) {
        local_C[i][j] = 0;
        for (int k = 0; k < k_size; ++k) {
            local_C[i][j] += local_A[i][k] * B[k][j];
        }
    }
}

MPI_Gather(local_C.data()->data(), rows_per_process * cols, MPI_INT, C.data()->data(), rows_per_process * cols, MPI_INT, 0, MPI_COMM_WORLD);
}

void matrix_mult_col_split(const vector<vector<int>> &A, const vector<vector<int>> &B, vector<vector<int>> &C, int rank, int size) {
// Реализуйте алгоритм умножения матриц с разбиением по столбцам
    int rows = A.size();
    int cols = B[0].size();
    int k_size = B.size();

    int cols_per_process = cols / size;

    vector<vector<int>> local_B(k_size, vector<int>(cols_per_process));
    vector<vector<int>> local_C(rows, vector<int>(cols_per_process));

    MPI_Scatter(B.data()->data(), k_size * cols_per_process, MPI_INT, local_B.data()->data(), k_size * cols_per_process, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(const_cast<int*>(A.data()->data()), rows * k_size, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols_per_process; ++j) {
            local_C[i][j] = 0;
            for (int k = 0; k < k_size; ++k) {
                local_C[i][j] += A[i][k] * local_B[k][j];
            }
        }
    }

    MPI_Gather(local_C.data()->data(), rows * cols_per_process, MPI_INT, C.data()->data(), rows * cols_per_process, MPI_INT, 0, MPI_COMM_WORLD);
}

void matrix_mult_block_split(const vector<vector<int>> &A, const vector<vector<int>> &B, vector<vector<int>> &C, int rank, int size) {
// Реализуйте алгоритм умножения матриц с разбиением по блокам
int rows = A.size();
int cols = B[0].size();
int k_size = B.size();

    int sqrt_size = sqrt(size);
    int block_rows = rows / sqrt_size;
    int block_cols = cols / sqrt_size;

    vector<vector<int>> local_A(block_rows, vector<int>(k_size));
    vector<vector<int>> local_B(k_size, vector<int>(block_cols));
    vector<vector<int>> local_C(block_rows, vector<int>(block_cols));

    MPI_Comm cart_comm;
    int dims[2] = {sqrt_size, sqrt_size};
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    int row_rank, col_rank;
    MPI_Comm row_comm;
    MPI_Comm col_comm;
    int row_dims[2] = {0, 1};
    MPI_Cart_sub(cart_comm, row_dims, &row_comm);
    int col_dims[2] = {1, 0};
    MPI_Cart_sub(cart_comm, col_dims, &col_comm);
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_rank(col_comm, &col_rank);

    int sendcounts[size], displs[size];
    for (int i = 0; i < sqrt_size; ++i) {
        for (int j = 0; j < sqrt_size; ++j) {
            sendcounts[i * sqrt_size + j] = 1;
            displs[i * sqrt_size + j] = i * block_rows * k_size + j * block_cols;
        }
    }

    MPI_Datatype block_type;
    MPI_Type_vector(block_rows, block_cols, cols, MPI_INT, &block_type);
    MPI_Type_commit(&block_type);

    MPI_Scatterv(A.data()->data(), sendcounts, displs, block_type, local_A.data()->data(), block_rows * k_size, MPI_INT, 0, col_comm);
    MPI_Scatterv(B.data()->data(), sendcounts, displs, block_type, local_B.data()->data(), k_size * block_cols, MPI_INT, 0, row_comm);

    for (int i = 0; i < block_rows; ++i) {
        for (int j = 0; j < block_cols; ++j) {
            local_C[i][j] = 0;
            for (int k = 0; k < k_size; ++k) {
                local_C[i][j] += local_A[i][k] * local_B[k][j];
            }
        }
    }

    MPI_Gatherv(local_C.data()->data(), block_rows * block_cols, MPI_INT, C.data()->data(), sendcounts, displs, block_type, 0, cart_comm);

    MPI_Type_free(&block_type);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);
}

bool check_matrix_mult(const vector<vector<int>> &A, const vector<vector<int>> &B, const vector<vector<int>> &C) {
    int rows = A.size();
    int cols = B[0].size();
    int k_size = B.size();

    vector<vector<int>> result(rows, vector<int>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = 0;
            for (int k = 0; k < k_size; ++k) {
                result[i][j]+= A[i][k] * B[k][j];
            }
        }
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (result[i][j] != C[i][j]) {
                return false;
            }
        }
    }

    return true;
}

