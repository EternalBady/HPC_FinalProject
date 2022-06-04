/*
	处理一维传热方程；
	这里petsc 的scalar和real区别还没有很清晰，统一先用scalar
*/
#include <petsc.h>
#include <math.h>
#define PI  3.14159265358979323846

int main(int argc, char **argv){

	PetscMPIInt		rank;
	PetscInt		i, j, size, iteration_num, Istart, Iend, col[3];
	PetscScalar		one = 1.0, dx = 0.01, dt = 0.001, 
					rho = 1.0, c = 1.0, l = 1.0, k = 1.0, lambda, gamma,
					value[3], val, fval;
	Vec				u_last, u_now, f;
	Mat 			A;
	//KSP				ksp;
	
	PetscCall(PetscInitialize(&argc,&argv,(char*)0,NULL));
	PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

	/*从命令行指定，dx, dt, l, rho, c的大小*/
	PetscOptionsGetScalar(NULL,NULL,"-dx",&dx,NULL);
	PetscOptionsGetScalar(NULL,NULL,"-dt",&dt,NULL);
	PetscOptionsGetScalar(NULL,NULL,"-l",&l,NULL);
	PetscOptionsGetScalar(NULL,NULL,"-rho",&rho,NULL);
	PetscOptionsGetScalar(NULL,NULL,"-c",&c,NULL);

	/*根据dx, dy 来获取vec,mat,以及迭代次数*/
	size 		  = 1/dx+1;
	iteration_num = 1/dt+1;
	lambda   	  = k*dt/(rho*c*dx*dx);
	gamma 		  = dt/(rho*c);
	/*初始化向量*/
	PetscCall(VecCreate(PETSC_COMM_WORLD,&u_last));
	PetscCall(VecSetSizes(u_last,PETSC_DECIDE,size));
	PetscCall(VecSetFromOptions(u_last));
	PetscCall(VecDuplicate(u_last,&u_now));
	PetscCall(VecDuplicate(u_now,&f));

	/*设置u_0*/
	PetscCall(VecGetOwnershipRange(u_last, &Istart, &Iend));
	for (i=Istart; i<Iend; i++) {
		val = exp(i*dx);
		PetscCall(VecSetValues(u_last,1,&i,&val,INSERT_VALUES));
	}
	PetscCall(VecSetValue(u_last,0,0.0,INSERT_VALUES));
	PetscCall(VecSetValue(u_last,size-1,0.0,INSERT_VALUES));
	PetscCall(VecAssemblyBegin(u_last));
    PetscCall(VecAssemblyEnd(u_last));
	PetscCall(VecView(u_last,PETSC_VIEWER_STDOUT_WORLD));

	/*设置f*/
	PetscCall(VecGetOwnershipRange(f, &Istart, &Iend));
	for (i=Istart; i<Iend; i++) {
		val = gamma * sin(l*i*dx*PI);
		PetscCall(VecSetValues(f,1,&i,&val,INSERT_VALUES));
	}
	
	PetscCall(VecAssemblyBegin(f));
    PetscCall(VecAssemblyEnd(f));
	// PetscCall(VecView(f,PETSC_VIEWER_STDOUT_WORLD));


	/*初始化矩阵A, 预分配一个三对角矩阵的大小*/
	PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  	PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,size,size));
  	PetscCall(MatSetFromOptions(A));
  	PetscCall(MatMPIAIJSetPreallocation(A,3,PETSC_NULL,3,PETSC_NULL));
	PetscCall(MatSetUp(A));

	PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));

	if (!Istart) {
        Istart = 1;
        i      = 0; col[0] = 0; col[1] = 1; value[0] = one+2.0*lambda; value[1] = -lambda;
        PetscCall(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
    }
    
    if (Iend == size) {
        Iend = size-1;
        i    = size-1; col[0] = size-2; col[1] = size-1; value[0] = -lambda; value[1] = 1.0+2.0*lambda;
        PetscCall(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
    }

    /* Set entries corresponding to the mesh interior */
    value[0] = -lambda; value[1] = one+2.0*lambda; value[2] = -lambda;
    for (i=Istart; i<Iend; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        PetscCall(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
    }

	PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  	PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
	// PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
	

	/*这部分后面写一个函数放到.h 里面模块化一下*/
	/*显示方法*/
	// for(i=0;i<iteration_num-1;i++){
	// 	for(j=1;j<size-1;j++){
	// 		col[0] = j-1; col[1] = j; col[2] = j+1;
	// 		PetscCall(VecGetValues(u_last, 1, &col[0], &val));
	// 		PetscPrintf(PETSC_COMM_WORLD,"size =  %g\n", value);
	// 	// val = lambda * () 
	// 	}
		

	// 	// PetscCall(VecSetValue(u_last,0,0.0,INSERT_VALUES));
	// 	// PetscCall(VecSetValue(u_last,size-1,0.0,INSERT_VALUES));
	// 	// PetscCall(VecAssemblyBegin(u_last));
	// 	// PetscCall(VecAssemblyEnd(u_last));
	// }
	for(i=0;i<iteration_num-1;i++){
		for(j=1;j<size-1;j++){
				col[0] = j-1; col[1] = j; col[2] = j+1;
				PetscCall(VecGetValues(u_last, 1, &col[0], &value[0]));
				PetscCall(VecGetValues(u_last, 1, &col[1], &value[1]));
				PetscCall(VecGetValues(u_last, 1, &col[2], &value[2]));
				PetscCall(VecGetValues(f, 1, &col[1], &fval));

				val = lambda * (value[0] + value[2])+(1-2*lambda) * value[1] + gamma * fval;
				PetscCall(VecSetValues(u_now,1,&j,&val,INSERT_VALUES));
			
		}
		PetscCall(VecSetValue(u_last,0,0.0,INSERT_VALUES));
		PetscCall(VecSetValue(u_last,size-1,0.0,INSERT_VALUES));
		PetscCall(VecAssemblyBegin(u_last));
		PetscCall(VecAssemblyEnd(u_last));

		PetscCall(VecCopy(u_now,u_last));
	}
	PetscCall(VecView(u_now,PETSC_VIEWER_STDOUT_WORLD));
	/*隐式方法*/



	// PetscCall(VecView(u_last,PETSC_VIEWER_STDOUT_WORLD));
	PetscPrintf(PETSC_COMM_WORLD,"size =  %D\n", size);

	/*程序执行完之后再进行析构和终结处理*/
	PetscCall(VecDestroy(&u_last));
	PetscCall(VecDestroy(&u_now));
	PetscCall(VecDestroy(&f));
	PetscCall(MatDestroy(&A));
	PetscCall(PetscFinalize());
	return 0;
}
