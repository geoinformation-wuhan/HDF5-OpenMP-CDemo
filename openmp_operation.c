#include <omp.h>
#include <stdio.h>
#include <hdf5.h>
#include <stdlib.h>
#include <time.h>

// 配置参数
#define MATRIX_SIZE 10000        // 矩阵大小 (2000x2000 = 400万个double元素，约32MB)
#define CHUNK_SIZE 500          // 数据块大小
#define NUM_DATASETS 10          // 数据集数量

// matrix calculation(matrix multiplication)
void matrix_multiply_parallel(double *A, double *B, double *C, int n) {
    printf("  [Parallel] Matrix Multiplication...\n");
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i*n + j] = 0.0;
            for (int k = 0; k < n; k++)
                C[i*n + j] += A[i*n + k] * B[k*n + j];
        }
    }
}
void matrix_multiply_serial(double *A, double *B, double *C, int n) {
    printf("  [Serial] Matrix Multiplication...\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i*n + j] = 0.0;
            for (int k = 0; k < n; k++)
                C[i*n + j] += A[i*n + k] * B[k*n + j];
        }
    }
}

//matrix data init
void init_matrix_parallel(double *matrix, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        unsigned int seed = i + omp_get_thread_num();
        for (int j = 0; j < n; j++) {
            matrix[i*n + j] = (double)rand_r(&seed) / RAND_MAX;
        }
    }
}
void init_matrix_serial(double *matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i*n + j] = (double)rand() / RAND_MAX;
        }
    }
}

void parallel_write_hdf5(const char* filename, double **matrices, int n, int num_matrices) {
/**
 * 并行写入HDF5文件
 * 演示：将大型矩阵数据并行写入多个数据集
 * 
 * @param filename 文件名
 * @param matrices 矩阵数组指针
 * @param n 矩阵维度
 * @param num_matrices 矩阵数量
 */
    printf("  [Parallel] Create HDF5 file and write %d %dx%d matrices...\n", num_matrices, n, n);
    
    hid_t file_id, dataspace_id, dataset_ids[num_matrices];
    herr_t status;
    hsize_t dims[2] = {n, n};
    char dataset_names[num_matrices][50];
    
    // 创建HDF5文件
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        printf("Error: Failed to create HDF5 file\n");
        return;
    }
    
    // 创建数据空间（所有数据集共享相同的数据空间）
    dataspace_id = H5Screate_simple(2, dims, NULL);
    for (int i = 0; i < num_matrices; i++)
        sprintf(dataset_names[i], "/matrix_%d", i);// 准备数据集名称
    #pragma omp parallel for
    for (int i = 0; i < num_matrices; i++) {
        dataset_ids[i] = H5Dcreate(file_id, dataset_names[i], H5T_IEEE_F64LE, 
                                  dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        
        if (dataset_ids[i] >= 0) {
            // 并行写入数据
            status = H5Dwrite(dataset_ids[i], H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
                             H5P_DEFAULT, matrices[i]);
            printf("    Thread %d: completed matrix %d\n", omp_get_thread_num(), i);
            H5Dclose(dataset_ids[i]);
        }
    }
    
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
    
    printf("  [Parallel] Write %d %dx%d matrices to HDF5 file %s\n", num_matrices, n, n, filename);
}

/**
 * 串行写入HDF5文件
 * 用于性能对比
 */
void serial_write_hdf5(const char* filename, double **matrices, int n, int num_matrices) {
    printf("  [Serial] Create HDF5 file and write %d %dx%d matrices...\n", num_matrices, n, n);
    
    hid_t file_id, dataspace_id, dataset_id;
    herr_t status;
    hsize_t dims[2] = {n, n};
    char dataset_name[50];
    
    // 创建HDF5文件
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        printf("Error: Failed to create HDF5 file\n");
        return;
    }
    
    // 创建数据空间
    dataspace_id = H5Screate_simple(2, dims, NULL);
    
    // 串行创建和写入数据集
    for (int i = 0; i < num_matrices; i++) {
        sprintf(dataset_name, "/matrix_%d", i);
        
        // 创建数据集
        dataset_id = H5Dcreate(file_id, dataset_name, H5T_IEEE_F64LE, dataspace_id,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dataset_id < 0) {
            printf("Error: Failed to create dataset %s\n", dataset_name);
            continue;
        }
        
        // 写入数据
        status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
                         H5P_DEFAULT, matrices[i]);
        if (status < 0) {
            printf("Error: Failed to write dataset %s\n", dataset_name);
        } else {
            printf("    Completed writing matrix %d\n", i);
        }
        
        H5Dclose(dataset_id);
    }
    
    // 关闭资源
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
    
    printf("  [Serial] Write %d %dx%d matrices to HDF5 file %s\n", num_matrices, n, n, filename);
}

/**
 * 并行读取HDF5文件中的矩阵数据
 * 演示：使用超平面选择(hyperslab)并行读取数据块
 * 
 * @param filename 文件名
 * @param matrices 用于存储读取数据的矩阵数组
 * @param n 矩阵维度
 * @param num_matrices 矩阵数量
 * @param chunk_size 每次读取的行数
 */
void parallel_read_hdf5(const char* filename, double **matrices, int n, 
                       int num_matrices, int chunk_size) {
    printf("  [Parallel] Read %d %dx%d matrices from HDF5 file (chunk size: %d)...\n", 
           num_matrices, n, n, chunk_size);
    
    hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

    #pragma omp parallel for
    for (int i = 0; i < num_matrices; i++) {
        char dataset_name[50];
        sprintf(dataset_name, "/matrix_%d", i);
        
        hid_t dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
        if (dataset_id >= 0) {
            herr_t status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
                                   H5P_DEFAULT, matrices[i]);
            H5Dclose(dataset_id);
            
            if (status >= 0) {
                printf("    Thread %d: Parallel read matrix %d\n", omp_get_thread_num(), i);
            }
        }
    }
    
    H5Fclose(file_id);
    printf("  [Parallel] Finish reading.\n");
}

/**
 * 串行读取HDF5文件
 * 用于性能对比
 */
void serial_read_hdf5(const char* filename, double **matrices, int n, int num_matrices) {
    printf("  [Serial] Read %d %dx%d matrices from HDF5 file...\n", num_matrices, n, n);
    
    hid_t file_id, dataset_id;
    herr_t status;
    char dataset_name[50];
    
    // 打开HDF5文件
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        printf("Error: Failed to open HDF5 file\n");
        return;
    }
    
    // 串行读取每个数据集
    for (int i = 0; i < num_matrices; i++) {
        sprintf(dataset_name, "/matrix_%d", i);
        
        // 打开数据集
        dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
        if (dataset_id < 0) {
            printf("Error: Failed to open dataset %s\n", dataset_name);
            continue;
        }
        
        // 读取整个数据集
        status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
                        H5P_DEFAULT, matrices[i]);
        if (status < 0) {
            printf("Error: Failed to read dataset %s\n", dataset_name);
        } else {
            printf("    Completed reading matrix %d\n", i);
        }
        
        H5Dclose(dataset_id);
    }
    
    H5Fclose(file_id);
    printf("  [Serial] Finish reading.\n");
}

/**
 * 验证矩阵数据的正确性
 * 简单检查：计算所有元素的和
 */
double verify_matrix(double *matrix, int n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n * n; i++) {
        sum += matrix[i];
    }
    return sum;
}
void print_performance_stats(const char* operation, double parallel_time, 
                           double serial_time, double data_size_mb) {
    printf("\n=== %s 性能统计 ===\n", operation);
    printf("数据大小: %.2f MB\n", data_size_mb);
    printf("并行时间: %.4f 秒\n", parallel_time);
    printf("串行时间: %.4f 秒\n", serial_time);
    printf("加速比: %.2fx\n", serial_time / parallel_time);
    printf("并行效率: %.2f%%\n", (serial_time / parallel_time) / omp_get_max_threads() * 100);
    printf("=============================\n\n");
}

int main(void) {
    // 性能计时变量
    double start_time, end_time;
    double parallel_time, serial_time;
    
    // 计算数据大小
    double matrix_size_mb = (double)(MATRIX_SIZE * MATRIX_SIZE * sizeof(double)) / (1024 * 1024);
    double total_data_mb = matrix_size_mb * NUM_DATASETS;
    
    printf("=== OpenMP + HDF5 并行计算演示程序 ===\n");
    printf("配置信息：\n");
    printf("  矩阵大小: %dx%d\n", MATRIX_SIZE, MATRIX_SIZE);
    printf("  数据集数量: %d\n", NUM_DATASETS);
    printf("  单个矩阵大小: %.2f MB\n", matrix_size_mb);
    printf("  总数据大小: %.2f MB\n", total_data_mb);
    printf("  OpenMP最大线程数: %d\n", omp_get_max_threads());
    printf("  数据块大小: %d 行\n\n", CHUNK_SIZE);
    
    // 分配内存
    printf("正在分配内存...\n");
    double **matrices = (double**)malloc(NUM_DATASETS * sizeof(double*));
    double **matrices_copy = (double**)malloc(NUM_DATASETS * sizeof(double*));
    
    for (int i = 0; i < NUM_DATASETS; i++) {
        matrices[i] = (double*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
        matrices_copy[i] = (double*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
        
        if (!matrices[i] || !matrices_copy[i]) {
            printf("内存分配失败！\n");
            return -1;
        }
    }
    
    // === 1. 矩阵初始化性能比较 ===
    printf("1. 矩阵初始化性能比较\n");
    
    // 并行初始化
    start_time = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < NUM_DATASETS; i++) {
        init_matrix_parallel(matrices[i], MATRIX_SIZE);
    }
    end_time = omp_get_wtime();
    parallel_time = end_time - start_time;
    
    // 串行初始化
    start_time = omp_get_wtime();
    for (int i = 0; i < NUM_DATASETS; i++) {
        init_matrix_serial(matrices_copy[i], MATRIX_SIZE);
    }
    end_time = omp_get_wtime();
    serial_time = end_time - start_time;
    
    print_performance_stats("矩阵初始化", parallel_time, serial_time, total_data_mb);
    
    // === 2. HDF5文件写入性能比较 ===
    printf("2. HDF5文件写入性能比较\n");
    
    // 并行写入
    start_time = omp_get_wtime();
    parallel_write_hdf5("parallel_data.h5", matrices, MATRIX_SIZE, NUM_DATASETS);
    end_time = omp_get_wtime();
    parallel_time = end_time - start_time;
    
    // 串行写入
    start_time = omp_get_wtime();
    serial_write_hdf5("serial_data.h5", matrices, MATRIX_SIZE, NUM_DATASETS);
    end_time = omp_get_wtime();
    serial_time = end_time - start_time;
    
    print_performance_stats("HDF5文件写入", parallel_time, serial_time, total_data_mb);
    
    // === 3. HDF5文件读取性能比较 ===
    printf("3. HDF5文件读取性能比较\n");
    
    // 清空内存用于验证读取结果
    for (int i = 0; i < NUM_DATASETS; i++) {
        memset(matrices[i], 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
        memset(matrices_copy[i], 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    }
    
    // 并行读取
    start_time = omp_get_wtime();
    parallel_read_hdf5("parallel_data.h5", matrices, MATRIX_SIZE, NUM_DATASETS, CHUNK_SIZE);
    end_time = omp_get_wtime();
    parallel_time = end_time - start_time;
    
    // 串行读取
    start_time = omp_get_wtime();
    serial_read_hdf5("serial_data.h5", matrices_copy, MATRIX_SIZE, NUM_DATASETS);
    end_time = omp_get_wtime();
    serial_time = end_time - start_time;
    
    print_performance_stats("HDF5文件读取", parallel_time, serial_time, total_data_mb);
    
    // === 4. 数据验证 ===
    printf("4. 数据完整性验证\n");
    for (int i = 0; i < NUM_DATASETS; i++) {
        double sum1 = verify_matrix(matrices[i], MATRIX_SIZE);
        double sum2 = verify_matrix(matrices_copy[i], MATRIX_SIZE);
        printf("  矩阵 %d: 并行读取校验和 = %.6f, 串行读取校验和 = %.6f\n", i, sum1, sum2);
    }
    
    // 释放内存
    printf("\n清理内存资源...\n");
    for (int i = 0; i < NUM_DATASETS; i++) {
        free(matrices[i]);
        free(matrices_copy[i]);
    }
    free(matrices);
    free(matrices_copy);
    
    printf("程序执行完成！生成的文件：\n");
    printf("  - parallel_data.h5 (并行写入)\n");
    printf("  - serial_data.h5 (串行写入)\n");
    printf("建议使用 'h5ls -v filename.h5' 查看文件结构\n");
    
    return 0;
}