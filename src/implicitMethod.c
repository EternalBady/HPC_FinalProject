/*
    隐式方法处理一维传热方程
*/
#include <petsc.h>
#include <petscviewerhdf5.h>
#include <math.h>

#define PI acos(-1)
#define FILE "implicitMethod.h5"

int main(int argc, char **argv)
{
    PetscMPIInt rank;
    PetscErrorCode ierr;
    // size就是从x的划分格数
    // iteration_num是迭代的次数，
    // restart 是指定是否重启，默认为0即不重启
    PetscInt i, size = 100, iteration_num, Istart, Iend, col[3] = {0, 1, 2}, 
             restart = 0, it = 0, testing_mode = 0;

    // dx 空间步长, dt 时间步长, t为当前的时刻
    // lambda=CFL 就是CFL, gamma 则为报告分析中简化计算的两个中间量
    PetscScalar dx, dt = 0.00001, error=0.0,
                rho = 1.0, c = 1.0, l = 1.0, k = 1.0,
                lambda, gamma, value[3], val, t = 0.0;

    // u_last 迭代过程中的上一次的u, u_now 迭代过程中当前的u, u_a是解析解;
    // f 为公式中的f, A 为迭代中的矩阵, temp为暂存向量
    Vec u_last, u_now, f, u_a, temp, b;
    Mat A;
    KSP ksp;
    // 计时器
    PetscLogDouble begin, end;
    PetscViewer viewer;

    ierr = PetscInitialize(&argc, &argv, (char *)0, NULL);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);

    /*Optional参数, 从命令行指定，dx, dt, l, rho, c的大小*/
    ierr = PetscOptionsGetScalar(NULL, NULL, "-dx", &dx, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetScalar(NULL, NULL, "-dt", &dt, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetScalar(NULL, NULL, "-l", &l, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetScalar(NULL, NULL, "-rho", &rho, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetScalar(NULL, NULL, "-c", &c, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-restart", &restart, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-size", &size, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL, NULL, "-testing_mode", &testing_mode, NULL);CHKERRQ(ierr);

    /*初始化向量u_last, u_now, f, 解析解u_a, 暂存向量temp*/
    ierr = VecCreate(PETSC_COMM_WORLD, &f);CHKERRQ(ierr);
    ierr = VecSetSizes(f, PETSC_DECIDE, size + 1);CHKERRQ(ierr);
    ierr = VecSetFromOptions(f);CHKERRQ(ierr);
    ierr = VecDuplicate(f, &u_now);CHKERRQ(ierr);
    ierr = VecDuplicate(f, &u_last);CHKERRQ(ierr);
    ierr = VecDuplicate(f, &u_a);CHKERRQ(ierr);
    ierr = VecDuplicate(f, &b);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_WORLD, &temp);CHKERRQ(ierr);
    ierr = VecSetSizes(temp, PETSC_DECIDE, 3);CHKERRQ(ierr);
    ierr = VecSetFromOptions(temp);CHKERRQ(ierr);

    /*初始化矩阵A 大小: (size+1)*(size+1), 预分配一个三对角矩阵的大小*/
    ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, size + 1, size + 1);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    //这里都设置为3, 因为一行最多的元素也就是3, 不会超过这个数；
    ierr = MatMPIAIJSetPreallocation(A, 3, PETSC_NULL, 3, PETSC_NULL);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);

    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, A, A);CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

    /*如果需要重启, 那么从FILE 中读取文件，如果不需要则执行下面的操作*/
    if (restart)
    {
        /* 用viewer打开FILE, 然后VecLoad两个存储的向量, 最后析构viewer */
        ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, FILE, FILE_MODE_READ, &viewer);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject) temp, "implicit-temp");CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject) u_last, "implicit-vector");CHKERRQ(ierr);
        ierr = VecLoad(temp, viewer);CHKERRQ(ierr);
        ierr = VecLoad(u_last, viewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

        /* 对temp所存的变量进行获取 */
        ierr = VecGetValues(temp, 3, col, value);CHKERRQ(ierr);
        it = value[0];
        size = value[1];
        dt = value[2];
    }
    else
    {
        it = 0;
        dx = 1.0 / size;
        /* 第一处计时, 计算设置向量和矩阵所花的时间 */
        ierr = PetscTime(&begin);CHKERRQ(ierr);
        /*设置u_0*/
        ierr = VecGetOwnershipRange(u_last, &Istart, &Iend);CHKERRQ(ierr);
        for (i = Istart; i < Iend; i++)
        {
            val = exp(i * dx); //题目给定的初始条件
            ierr = VecSetValues(u_last, 1, &i, &val, INSERT_VALUES);CHKERRQ(ierr);
        }
        // 设置边界条件即0 和 size 处为0
        ierr = VecSetValue(u_last, 0, 0.0, INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecSetValue(u_last, size, 0.0, INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecAssemblyBegin(u_last);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(u_last);CHKERRQ(ierr);
        // ierr = VecView(u_last,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    /*根据size来获取dx,以及迭代次数, 计算简化计算用的 lambda和gamma*/
    dx = 1.0 / size;
    iteration_num = 1 / dt + 1;
    lambda = k * dt / (rho * c * dx * dx);
    gamma = dt / (rho * c);

    if (lambda > 0.5)
    {
        //如果CFL > 0.5, 则不可能收敛, 直接退出程序
        ierr = PetscPrintf(PETSC_COMM_WORLD, "CFL =  %g, 程序无法收敛请重新指定输入参数！\n", lambda);CHKERRQ(ierr);
        exit(0);
    }


    /*设置f*/
    ierr = VecGetOwnershipRange(f, &Istart, &Iend);CHKERRQ(ierr);
    for (i = Istart; i < Iend; i++)
    {
        val = sin(l * i * dx * PI); //题目给定的初始条件
        ierr = VecSetValues(f, 1, &i, &val, INSERT_VALUES);CHKERRQ(ierr);
    }

    ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
    // ierr = VecView(f,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    /* 设置三对角矩阵 */
    ierr = MatGetOwnershipRange(A, &Istart, &Iend);CHKERRQ(ierr);

    if (!Istart)
    {
        Istart = 1;
        i = 0;
        col[0] = 0;
        col[1] = 1;
        value[0] = 1.0 + 2.0 * lambda;
        value[1] = lambda;
        ierr = MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES);CHKERRQ(ierr);
    }

    if (Iend == size + 1)
    {
        Iend = size;
        i = size;
        col[0] = size - 1;
        col[1] = size;
        value[0] = -lambda;
        value[1] = 1.0 - 2.0 * lambda;
        ierr = MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES);CHKERRQ(ierr);
    }

    /* 除了两行特判之外其他都符合一行三个元素 */
    value[0] = -lambda;
    value[1] = 1.0 + 2.0 * lambda;
    value[2] = -lambda;
    for (i = Istart; i < Iend; i++)
    {
        col[0] = i - 1;
        col[1] = i;
        col[2] = i + 1;
        ierr = MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES);CHKERRQ(ierr);
    }

    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    // ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    if (!restart)
    {   
        // 这里如果是非重启模式才会对程序构造向量和矩阵进行计时;
        ierr = PetscTime(&end);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Assembly Time = %g\n", end - begin);CHKERRQ(ierr);
    }

    /* 打印程序的参数，查看是否有错 */
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Lambda =  %g, gamma = %g\n", lambda, gamma);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "it =  %d, size = %d, dt =  %g\n", it, size+1, dt);CHKERRQ(ierr);

    /* 造一个解析解, 此处不参与计时 */
    ierr = VecGetOwnershipRange(u_a, &Istart, &Iend);CHKERRQ(ierr);
    for (i = Istart; i < Iend; i++)
    {
        val = (PetscScalar) (sin(PI*dx*i*l) - sin(l*PI)*dx*i)/(PI*PI*l*l); // 推导得出的解析解
        ierr = VecSetValues(u_a, 1, &i, &val, INSERT_VALUES);CHKERRQ(ierr);
    }

    ierr = VecAssemblyBegin(u_a);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(u_a);CHKERRQ(ierr);
    // ierr = VecView(u_a,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    /* 第二处计时, 计算迭代所花的时间 */
    ierr = PetscTime(&begin);CHKERRQ(ierr);

    PetscLogDouble t1, t2;
    PetscScalar    t_iter, t_write;
    ierr = PetscTime(&t1);CHKERRQ(ierr);
    t += dt;
    it ++;

    ierr = VecWAXPY(b, gamma, f, u_last);CHKERRQ(ierr);
    ierr = KSPSolve(ksp, b, u_now);CHKERRQ(ierr);

    ierr = VecSetValue(u_now, 0, 0.0, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(u_now, size, 0.0, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(u_now);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(u_now);CHKERRQ(ierr);
    // 将当前结果赋给上一次的vector
    ierr = VecCopy(u_now, u_last);CHKERRQ(ierr);
    ierr = PetscTime(&t2);CHKERRQ(ierr);
    t_iter = t2-t1;

    ierr = PetscTime(&t1);CHKERRQ(ierr);
    value[0] = it;
    value[1] = size;
    value[2] = dt;
    col[0] = 0;
    col[1] = 1;
    col[2] = 2;
    ierr = VecSetValues(temp, 3, col, value, INSERT_VALUES);CHKERRQ(ierr);

    ierr = VecAssemblyBegin(temp);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(temp);CHKERRQ(ierr);

    /*利用viewer 存储temp和u_last*/
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, FILE, FILE_MODE_WRITE, &viewer);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) temp,   "implicit-temp");CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) u_last, "implicit-vector");CHKERRQ(ierr);
    ierr = VecView(temp, viewer);CHKERRQ(ierr);
    ierr = VecView(u_last, viewer);CHKERRQ(ierr);

    ierr = PetscTime(&t2);CHKERRQ(ierr);
    t_write = t2-t1;
    int n_step = (int)100 * t_write / t_iter;
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Write times = %d, t_iter=%g, t_write=%g\n", n_step, t_iter, t_write);CHKERRQ(ierr);

    // ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, FILE, FILE_MODE_WRITE, &viewer);CHKERRQ(ierr);
    for (; it < iteration_num; it++)
    {
        t += dt;

        ierr = VecWAXPY(b, gamma, f, u_last);CHKERRQ(ierr);
        ierr = KSPSolve(ksp, b, u_now);CHKERRQ(ierr);

        ierr = VecSetValue(u_now, 0, 0.0, INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecSetValue(u_now, size, 0.0, INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecAssemblyBegin(u_now);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(u_now);CHKERRQ(ierr);
        // 将当前结果赋给上一次的vector
        ierr = VecCopy(u_now, u_last);CHKERRQ(ierr);

        /*
            每n_step次迭代保存一次断点, 这里需要保存的有
            1. it    : 迭代到哪一步;
            2. size  : 网格数量
            3. dt    : 时间步大小
            4. u_last: 当前迭代的结果,copy了u_now的结果, 存u_last 有利于和上面的读文件统一
            保存文件的耗时非常高, 在一些测试中会暂时移除
        */
        if (!(testing_mode)&&!(it % n_step))
        {
            value[0] = it;
            value[1] = size;
            value[2] = dt;
            col[0] = 0;
            col[1] = 1;
            col[2] = 2;
            ierr = VecSetValues(temp, 3, col, value, INSERT_VALUES);CHKERRQ(ierr);

            ierr = VecAssemblyBegin(temp);CHKERRQ(ierr);
            ierr = VecAssemblyEnd(temp);CHKERRQ(ierr);

            /*利用viewer 存储temp和u_last*/
            // ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
            // ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, FILE, FILE_MODE_WRITE, &viewer);CHKERRQ(ierr);
            ierr = PetscObjectSetName((PetscObject) temp,   "implicit-temp");CHKERRQ(ierr);
            ierr = PetscObjectSetName((PetscObject) u_last, "implicit-vector");CHKERRQ(ierr);
            ierr = VecView(temp, viewer);CHKERRQ(ierr);
            ierr = VecView(u_last, viewer);CHKERRQ(ierr);
            // ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
        }
    }
    ierr = PetscTime(&end);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Iteration Time = %g\n", end - begin);CHKERRQ(ierr);

    // ierr = VecView(u_now, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    /*误差分析, 跟解析解对比*/
    PetscScalar x = 0.0, y = 0.0; // 用来暂存对比
    for (i = Istart; i < Iend; i++){
        ierr = VecGetValues(u_a, 1, &i, &x);CHKERRQ(ierr);
        ierr = VecGetValues(u_last, 1, &i, &y);CHKERRQ(ierr);
        if(fabs(x-y) > error){
            error = fabs(x-y);
        }   
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Error = %g\n", error);CHKERRQ(ierr);

    /*程序执行完之后再进行析构和终结处理*/
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = VecDestroy(&u_last);CHKERRQ(ierr);
    ierr = VecDestroy(&u_now);CHKERRQ(ierr);
    ierr = VecDestroy(&f);CHKERRQ(ierr);
    ierr = VecDestroy(&u_a);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
    ierr = VecDestroy(&temp);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
    ierr = PetscFinalize();CHKERRQ(ierr);
    return ierr;
}
