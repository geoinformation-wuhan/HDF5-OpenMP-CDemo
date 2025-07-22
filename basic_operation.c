#include <stdio.h>
#include <hdf5.h>

int main(void) {
    // 声明变量
    const int rank = 2;       /* 数据空间维度数 */
    const int nrow = 4;       /* 数据行数 */
    const int ncol = 6;       /* 数据列数 */
    hid_t file_id;      /* 文件标识符 */
    herr_t status;      /* 函数返回状态 */
    hid_t dataspace_id; /* 数据空间标识符 */
    hid_t dataset_id;   /* 数据集标识符 */
    hid_t attr_id;      /* 属性标识符 */
    hid_t attr_dataspace_id; /* 属性数据空间标识符 */
    hid_t group_id;     /* 组标识符 */
    hid_t subgroup_id;  /* 子组标识符 */
    hsize_t dims[rank];    /* 数据空间维度 */
    hsize_t attr_dims[1]; /* 属性数据空间维度 */
    
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
    
    printf("HDF5 file created successfully!\n\n");

    // === 创建组结构 ===
    printf("Creating group hierarchy...\n");
    
    // 1. 使用绝对路径创建主组
    group_id = H5Gcreate(file_id, "/DataGroup", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (group_id < 0) {
        printf("Error: Could not create main group.\n");
        H5Fclose(file_id);
        return -1;
    }
    printf("Created main group: /DataGroup\n");

    // 2. 使用绝对路径创建子组A
    subgroup_id = H5Gcreate(file_id, "/DataGroup/Matrices", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (subgroup_id < 0) {
        printf("Error: Could not create subgroup Matrices.\n");
        H5Gclose(group_id);
        H5Fclose(file_id);
        return -1;
    }
    printf("Created subgroup: /DataGroup/Matrices\n");
    H5Gclose(subgroup_id);

    // 3. 使用相对路径创建子组B（相对于DataGroup）
    subgroup_id = H5Gcreate(group_id, "Attributes", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (subgroup_id < 0) {
        printf("Error: Could not create subgroup Attributes.\n");
        H5Gclose(group_id);
        H5Fclose(file_id);
        return -1;
    }
    printf("Created subgroup: /DataGroup/Attributes (using relative path)\n");
    H5Gclose(subgroup_id);

    printf("Group hierarchy created successfully!\n\n");

    // === 在组中创建数据集 ===
    printf("Creating dataset in group...\n");

    // 设置数据空间维度
    dims[0] = nrow; 
    dims[1] = ncol;
    
    // 创建简单的数据空间
    // H5Screate_simple的参数：维度数，当前维度，最大维度（NULL表示固定大小）
    dataspace_id = H5Screate_simple(rank, dims, NULL);
    if (dataspace_id < 0) {
        printf("Error: Could not create dataspace.\n");
        H5Gclose(group_id);
        H5Fclose(file_id);
        return -1;
    }

    // 在Matrices子组中创建数据集
    dataset_id = H5Dcreate(file_id, "/DataGroup/Matrices/matrix_data", H5T_STD_I32BE, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
        printf("Error: Could not create dataset.\n");
        H5Sclose(dataspace_id);
        H5Gclose(group_id);
        H5Fclose(file_id);
        return -1;
    }

    printf("Dataset created in group: /DataGroup/Matrices/matrix_data\n");

    // 创建示例数据
    int data[nrow][ncol];
    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < ncol; j++) {
            data[i][j] = i * ncol + j + 1;  // 填充1到24的数字
        }
    }

    // 写入数据集
    status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                      H5P_DEFAULT, data);
    if (status < 0) {
        printf("Error: Could not write to dataset.\n");
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Gclose(group_id);
        H5Fclose(file_id);
        return -1;
    }

    printf("Data written to dataset successfully!\n\n");

    // === 创建属性 ===
    printf("Creating and writing attributes...\n");
    
    // 为属性创建数据空间（一维数组，包含2个整数）
    attr_dims[0] = 2;
    attr_dataspace_id = H5Screate_simple(1, attr_dims, NULL);
    if (attr_dataspace_id < 0) {
        printf("Error: Could not create attribute dataspace.\n");
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Gclose(group_id);
        H5Fclose(file_id);
        return -1;
    }

    // 创建属性（附加到数据集）
    attr_id = H5Acreate(dataset_id, "dimensions", H5T_STD_I32BE, attr_dataspace_id,
                        H5P_DEFAULT, H5P_DEFAULT);
    if (attr_id < 0) {
        printf("Error: Could not create attribute.\n");
        H5Sclose(attr_dataspace_id);
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Gclose(group_id);
        H5Fclose(file_id);
        return -1;
    }

    // 准备属性数据（存储矩阵维度）
    int attr_data[2] = {nrow, ncol};

    // 写入属性
    status = H5Awrite(attr_id, H5T_NATIVE_INT, attr_data);
    if (status < 0) {
        printf("Error: Could not write attribute.\n");
        H5Aclose(attr_id);
        H5Sclose(attr_dataspace_id);
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Gclose(group_id);
        H5Fclose(file_id);
        return -1;
    }

    printf("Attribute 'dimensions' written successfully!\n");

    // 关闭属性和属性数据空间
    H5Aclose(attr_id);
    H5Sclose(attr_dataspace_id);

    printf("\n=== Reading Data ===\n");

    // 从数据集读取数据
    int data_read[nrow][ncol];
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                     H5P_DEFAULT, data_read);
    if (status < 0) {
        printf("Error: Could not read from dataset.\n");
    } else {
        // 打印读取的数据
        printf("Matrix data from /DataGroup/Matrices/matrix_data:\n");
        for (int i = 0; i < nrow; i++) {
            for (int j = 0; j < ncol; j++) {
                printf("%3d ", data_read[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("Dividing the %dx%d matrix into four quarters and reading each separately...\n\n", nrow, ncol);
    
    // 获取数据集的文件数据空间
    hid_t file_dataspace_id = H5Dget_space(dataset_id);
    if (file_dataspace_id < 0) {
        printf("Error: Could not get dataset dataspace.\n");
    } else {
        // 定义四个象限的参数
        struct {
            const char* name;
            hsize_t offset[2];
            hsize_t count[2];
        } quarters[4] = {
            {"Top-Left",     {0, 0}, {nrow/2, ncol/2}},  // 左上角
            {"Top-Right",    {0, ncol/2}, {nrow/2, ncol/2}},  // 右上角
            {"Bottom-Left",  {nrow/2, 0}, {nrow/2, ncol/2}},  // 左下角
            {"Bottom-Right", {nrow/2, ncol/2}, {nrow/2, ncol/2}}  // 右下角
        };
        
        // 循环读取四个象限
        for (int q = 0; q < 4; q++) {
            printf("Reading %s quarter (offset: [%llu,%llu], size: [%llu,%llu]):\n", 
                   quarters[q].name, 
                   (unsigned long long)quarters[q].offset[0], 
                   (unsigned long long)quarters[q].offset[1],
                   (unsigned long long)quarters[q].count[0], 
                   (unsigned long long)quarters[q].count[1]);
            
            // 选择文件数据空间的子集（hyperslab selection）
            status = H5Sselect_hyperslab(file_dataspace_id, H5S_SELECT_SET,
                                        quarters[q].offset, NULL,  // stride为NULL表示使用默认值1
                                        quarters[q].count, NULL);  // block为NULL表示使用默认值1
            if (status < 0) {
                printf("Error: Could not select hyperslab for %s quarter.\n", quarters[q].name);
                continue;
            }
            
            // 为内存数据创建数据空间（必须与选择的元素数量匹配）
            hid_t mem_dataspace_id = H5Screate_simple(rank, quarters[q].count, NULL);
            if (mem_dataspace_id < 0) {
                printf("Error: Could not create memory dataspace for %s quarter.\n", quarters[q].name);
                continue;
            }
            
            // 验证内存和文件数据空间的元素数量是否匹配
            hssize_t file_points = H5Sget_select_npoints(file_dataspace_id);
            hssize_t mem_points = H5Sget_select_npoints(mem_dataspace_id);
            if (file_points != mem_points) {
                printf("Error: Memory and file dataspace sizes don't match (%lld vs %lld).\n",
                       (long long)mem_points, (long long)file_points);
                H5Sclose(mem_dataspace_id);
                continue;
            }
            
            // 分配内存来存储子集数据
            int quarter_size_0 = (int)quarters[q].count[0];
            int quarter_size_1 = (int)quarters[q].count[1];
            int quarter_data[quarter_size_0][quarter_size_1];
            
            // 读取子集数据
            status = H5Dread(dataset_id, H5T_NATIVE_INT, mem_dataspace_id, file_dataspace_id,
                            H5P_DEFAULT, quarter_data);
            if (status < 0) {
                printf("Error: Could not read %s quarter data.\n", quarters[q].name);
            } else {
                // 显示读取的子集数据
                for (int i = 0; i < quarter_size_0; i++) {
                    printf("  ");
                    for (int j = 0; j < quarter_size_1; j++) {
                        printf("%3d ", quarter_data[i][j]);
                    }
                    printf("\n");
                }
            }
            
            // 关闭内存数据空间
            H5Sclose(mem_dataspace_id);
            printf("\n");
        }
        
        // 关闭文件数据空间
        H5Sclose(file_dataspace_id);
    }

    printf("Subset reading demonstration completed!\n");

    // 读取属性
    printf("Reading attributes...\n");
    
    // 打开已存在的属性
    attr_id = H5Aopen(dataset_id, "dimensions", H5P_DEFAULT);
    if (attr_id < 0) {
        printf("Error: Could not open attribute.\n");
    } else {
        // 准备接收属性数据的数组
        int attr_read_data[2];
        
        // 读取属性数据
        status = H5Aread(attr_id, H5T_NATIVE_INT, attr_read_data);
        if (status < 0) {
            printf("Error: Could not read attribute.\n");
        } else {
            printf("Matrix dimensions from attribute: %d x %d\n", 
                   attr_read_data[0], attr_read_data[1]);
        }
        
        // 关闭属性
        H5Aclose(attr_id);
    }

    // 关闭数据集
    H5Dclose(dataset_id);
    printf("Dataset closed.\n");

    // 关闭数据空间
    H5Sclose(dataspace_id);
    printf("Dataspace closed.\n");
    
    // 关闭组
    H5Gclose(group_id);
    printf("Group closed.\n");
    
    // 关闭文件
    status = H5Fclose(file_id);
    if (status < 0) {
        printf("Error: Could not close HDF5 file.\n");
        return -1;
    }
    
    printf("HDF5 file closed successfully.\n");
    printf("\n=== Demo completed! ===\n");
    printf("Use 'h5dump example.h5' to view the file structure.\n");
    
    return 0;
}