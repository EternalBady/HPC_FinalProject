/*
    隐式方法处理一维传热方程；
*/
#include <petsc.h>
#include <math.h>
#define PI 3.14159265358979323846

int main(int argc, char **argv)
{

    //定义每次执行时的processor编号rank
    PetscMPIInt rank;
    // size就是从x的划分，
    // iteration_num是迭代的次数，
    // Istart和Iend是每个process里存放的矩阵或向量编号
    // col[] 是进行三对角矩阵设置时的索引数组。
    PetscInt i, size, iteration_num, Istart, Iend, col[3];

    // dx 空间步长, dt 时间步长
    // rho, c, l, k则是题目给定的参数，这里取为1
    // lambda, gamma 则为报告分析中简化计算的两个中间量
    // value[] 设置三对角矩阵时存储的三个值
    PetscScalar one = 1.0, dx = 0.01, dt = 0.00001,
                rho = 1.0, c = 1.0, l = 1.0, k = 1.0,
                lambda, gamma, value[3], val;
    // u_last 迭代过程中的上一次的u, u_now 迭代过程中当前的u;
    // f 为公式中的f, b为矩阵计算的中间量;
    // A 为迭代中的矩阵, ksp 线性方程组的求解器
    Vec u_last, u_now, f, b;
    Mat A;
    KSP ksp;
    //计时器
    PetscLogDouble begin, end;

    PetscCall(PetscInitialize(&argc, &argv, (char *)0, NULL));
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    /* Optional参数, 从命令行指定，dx, dt, l, rho, c的大小*/
    PetscOptionsGetScalar(NULL, NULL, "-dx", &dx, NULL);
    PetscOptionsGetScalar(NULL, NULL, "-dt", &dt, NULL);
    PetscOptionsGetScalar(NULL, NULL, "-l", &l, NULL);
    PetscOptionsGetScalar(NULL, NULL, "-rho", &rho, NULL);
    PetscOptionsGetScalar(NULL, NULL, "-c", &c, NULL);

    /*根据dx, dy 来获取vec,mat,以及迭代次数, 计算简化计算用的 lambda和gamma*/
    size = 1 / dx + 1;
    iteration_num = 1 / dt + 1;
    lambda = k * dt / (rho * c * dx * dx);
    gamma = dt / (rho * c);

    /*初始化向量u_last, u_now, f, b*/
    PetscCall(VecCreate(PETSC_COMM_WORLD, &u_last));
    PetscCall(VecSetSizes(u_last, PETSC_DECIDE, size));
    PetscCall(VecSetFromOptions(u_last));
    PetscCall(VecDuplicate(u_last, &u_now));
    PetscCall(VecDuplicate(u_now, &f));
    PetscCall(VecDuplicate(u_now, &b));

    /*初始化矩阵A 大小: size*size, 预分配一个三对角矩阵的大小*/
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, size, size));
    PetscCall(MatSetFromOptions(A));
    //这里都设置为3, 因为一行最多的元素也就是3, 不会超过这个数；
    PetscCall(MatMPIAIJSetPreallocation(A, 3, PETSC_NULL, 3, PETSC_NULL));
    PetscCall(MatSetUp(A));

    /* 第一处计时, 计算设置向量和矩阵所花的时间 */
    PetscCall(PetscTime(&begin));
    /*设置u_0*/
    PetscCall(VecGetOwnershipRange(u_last, &Istart, &Iend));
    for (i = Istart; i < Iend; i++)
    {
        val = exp(i * dx); //题目给定的初始条件
        PetscCall(VecSetValues(u_last, 1, &i, &val, INSERT_VALUES));
    }
    // 设置边界条件即0 和 size-1 处为0
    PetscCall(VecSetValue(u_last, 0, 0.0, INSERT_VALUES));
    PetscCall(VecSetValue(u_last, size - 1, 0.0, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(u_last));
    PetscCall(VecAssemblyEnd(u_last));
    // PetscCall(VecView(u_last,PETSC_VIEWER_STDOUT_WORLD));

    /*设置f*/
    PetscCall(VecGetOwnershipRange(f, &Istart, &Iend));
    for (i = Istart; i < Iend; i++)
    {
        val = sin(l * i * dx * PI); //题目给定的初始条件
        PetscCall(VecSetValues(f, 1, &i, &val, INSERT_VALUES));
    }

    PetscCall(VecAssemblyBegin(f));
    PetscCall(VecAssemblyEnd(f));
    // PetscCall(VecView(f,PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
    if (!Istart)
    {
        Istart = 1;
        i = 0;
        col[0] = 0;
        col[1] = 1;
        value[0] = one + 2.0 * lambda;
        value[1] = -lambda;
        PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
    }

    if (Iend == size)
    {
        Iend = size - 1;
        i = size - 1;
        col[0] = size - 2;
        col[1] = size - 1;
        value[0] = -lambda;
        value[1] = 1.0 + 2.0 * lambda;
        PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
    }

    /* Set entries corresponding to the mesh interior */
    value[0] = -lambda;
    value[1] = one + 2.0 * lambda;
    value[2] = -lambda;
    for (i = Istart; i < Iend; i++)
    {
        col[0] = i - 1;
        col[1] = i;
        col[2] = i + 1;
        PetscCall(MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES));
    }

    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    // PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

    /* 第一次计时结束打印出设置矩阵的运行时间 */
    PetscCall(PetscTime(&end));
    PetscPrintf(PETSC_COMM_WORLD, "Assembly Time = %g\n", end-begin);
    PetscPrintf(PETSC_COMM_WORLD, "Lambda =  %g, gamma = %g\n", lambda, gamma);

    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetTolerances(ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCall(KSPSetFromOptions(ksp));

    /*隐式方法*/
    /* 第二次计时开始, 计算迭代所花的时间 */
    PetscCall(PetscTime(&begin));
    for (i = 0; i < iteration_num; i++)
    {

        PetscCall(VecWAXPY(b, gamma, f, u_last));
        PetscCall(KSPSolve(ksp, b, u_now));

        /* 记得边界条件设置为0 */
        PetscCall(VecSetValue(u_now, 0, 0.0, INSERT_VALUES));
        PetscCall(VecSetValue(u_now, size - 1, 0.0, INSERT_VALUES));
        PetscCall(VecAssemblyBegin(u_now));
        PetscCall(VecAssemblyEnd(u_now));

        PetscCall(VecCopy(u_now, u_last));

        /* 每20次迭代保存一次断点 */
    }
    /* 第二次计时结束，打印出运行时间 */
    PetscCall(PetscTime(&end));
    PetscCall(VecView(u_now, PETSC_VIEWER_STDOUT_WORLD));
    PetscPrintf(PETSC_COMM_WORLD, "Iteration Time = %g\n", end-begin);

    /* 将最终结果输出到文件中 */

    // PetscCall(VecView(u_last,PETSC_VIEWER_STDOUT_WORLD));
    PetscPrintf(PETSC_COMM_WORLD, "size =  %D\n", size);

    /*程序执行完之后再进行析构和终结处理*/
    PetscCall(VecDestroy(&u_last));
    PetscCall(VecDestroy(&u_now));
    PetscCall(VecDestroy(&f));
    PetscCall(VecDestroy(&b));
    PetscCall(MatDestroy(&A));
    PetscCall(PetscFinalize());
    return 0;
}
