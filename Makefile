-include ../../../../petscdir.mk
PETSC_DIR=/home/bady/lib/petsc-3.16.6-opt

CFLAGS           = -O3
CPPFLAGS         = 
EXAMPLESC        = test1.c
MANSEC           = KSP
CLEANFILES       = rhs.vtk solution.vtk
NP               = 1
DIRS             = network amrex

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

