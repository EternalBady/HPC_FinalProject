/*
	处理一维传热方程；
	这里petsc 的scalar和real区别还没有很清晰，统一先用scalar
*/
#include <petscksp.h>
#include <time.h>

int main(int argc, char **argv){

	PetscMPIInt		rank;
	PetscInt		i, N;
	PetscScalar		one = 1.0, zero = 0.0, dx = 0.01, dt = 0.001,
					rho = 1.0, c = 1.0, l = 1.0, k = 1.0;
	Vec				x, y, z, u_last, u_now, f; //显式需要用到 u_(n+1),u_(n),u_(n-1)
	Mat 			A;
	MPI_Comm 		comm = PETSC_COMM_WORLD;
	KSP				ksp;
	
	PetscCall(PetscInitialize(&argc,&argv));
	PetscCallMPI(MPI_Comm_rank(comm,&rank));

	/*初始化向量*/

	PetscCall(VecAssemblyBegin(x));
    PetscCall(VecAssemblyEnd(x));


	/*初始化矩阵A, 预分配一个三对角矩阵的大小*/

	PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  	PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

	/*这部分后面写一个函数放到.h 里面模块化一下*/
	/*显示方法*/


	/*隐式方法*/


	/*程序执行完之后再进行析构和终结处理*/
	PetscCall(VecDestroy(&x));
	PetscCall(PetscFinalize());
	return 0;
}
