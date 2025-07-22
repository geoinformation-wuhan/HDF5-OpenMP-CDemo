#include <stdio.h>
#include <hdf5.h>

int main(void) {
    // 声明变量
    hid_t file_id;      /* 文件标识符 */
    herr_t status;      /* 函数返回状态 */
    hid_t dataspace_id; /* 数据空间标识符 */
    hid_t dataset_id;   /* 数据集标识符 */
    hsize_t dims[2];    /* 数据空间维度 */
    const int rank = 2;       /* 数据空间维度数 */
    const int nrow = 4;
    const int ncol = 6;
    
    printf("Creating an HDF5 file...\n");
    
    // 创建新的HDF5文件
    // H5F_ACC_TRUNC: 如果文件已存在，则删除当前内容；如不存在则创建新文件
    // H5P_DEFAULT: 使用默认文件创建属性列表
    // H5P_DEFAULT: 使用默认文件访问属性列表
    file_id = H5Fcreate("example.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        printf("Error: Could not create HDF5 file.\n");
        return -1;
    }
    
    printf("HDF5 file created successfully!\n");

    // 设置数据空间维度
    dims[0] = nrow; dims[1] = ncol;
    
    // 创建简单的数据空间
    // H5Screate_simple的参数：维度数，当前维度，最大维度（NULL表示固定大小）
    dataspace_id = H5Screate_simple(rank, dims, NULL);
    if (dataspace_id < 0) {
        printf("Error: Could not create dataspace.\n");
        H5Fclose(file_id);
        return -1;
    }

    printf("Dataspace created successfully!\n");

    // 创建数据集
    // 参数：文件ID，数据集名称，数据类型，数据空间ID，创建属性列表，
    //      链接创建属性列表，访问属性列表
    dataset_id = H5Dcreate(file_id, "/dset", H5T_STD_I32BE, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
        printf("Error: Could not create dataset.\n");
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
        return -1;
    }

    printf("Dataset created successfully!\n");

    // 创建示例数据
    int data[nrow][ncol];
    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < ncol; j++) {
            data[i][j] = i * 6 + j + 1;  // 填充1到24的数字
        }
    }

    // 写入数据集
    status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
    if (status < 0) {
        printf("Error: Could not write to dataset.\n");
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
        return -1;
    }

    printf("Data written to dataset successfully!\n");

    // 从数据集读取数据
    int data_read[nrow][ncol];
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                     H5P_DEFAULT, data_read);
    if (status < 0) {
        printf("Error: Could not read from dataset.\n");
        H5Dclose(dataset_id);
    }

    // 打印读取的数据
    printf("The data read from the dataset is:\n");
    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < ncol; j++) {
            printf("%d\t", data_read[i][j]);
        }
        printf("\n");
    }

    // 关闭数据空间
    status = H5Sclose(dataspace_id);
    if (status < 0) {
        printf("Error: Could not close dataspace.\n");
        H5Fclose(file_id);
        return -1;
    }
    
    // 终止对文件的访问
    status = H5Fclose(file_id);
    if (status < 0) {
        printf("Error: Could not close HDF5 file.\n");
        return -1;
    }
    
    printf("HDF5 file closed successfully.\n");
    return 0;
}