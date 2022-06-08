-include ../../../../petscdir.mk
PETSC_DIR=/home/bady/lib/petsc-3.16.6-opt

CFLAGS           = 
CPPFLAGS         = 
EXAMPLESC        = main.c, implicitMethod.c, test1.c, explicitMethod.c
MANSEC           = KSP
CLEANFILES       = implicitMethod.h5 explicitMethod.h5
NP               = 1
DIRS             = src

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

