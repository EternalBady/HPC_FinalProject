# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/bady/HPC_FinalProject/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bady/HPC_FinalProject/build

# Include any dependencies generated for this target.
include CMakeFiles/implicitMethod.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/implicitMethod.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/implicitMethod.dir/flags.make

CMakeFiles/implicitMethod.dir/implicitMethod.c.o: CMakeFiles/implicitMethod.dir/flags.make
CMakeFiles/implicitMethod.dir/implicitMethod.c.o: /home/bady/HPC_FinalProject/src/implicitMethod.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bady/HPC_FinalProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/implicitMethod.dir/implicitMethod.c.o"
	/home/bady/lib/mpich/bin/mpicc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/implicitMethod.dir/implicitMethod.c.o   -c /home/bady/HPC_FinalProject/src/implicitMethod.c

CMakeFiles/implicitMethod.dir/implicitMethod.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/implicitMethod.dir/implicitMethod.c.i"
	/home/bady/lib/mpich/bin/mpicc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/bady/HPC_FinalProject/src/implicitMethod.c > CMakeFiles/implicitMethod.dir/implicitMethod.c.i

CMakeFiles/implicitMethod.dir/implicitMethod.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/implicitMethod.dir/implicitMethod.c.s"
	/home/bady/lib/mpich/bin/mpicc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/bady/HPC_FinalProject/src/implicitMethod.c -o CMakeFiles/implicitMethod.dir/implicitMethod.c.s

# Object files for target implicitMethod
implicitMethod_OBJECTS = \
"CMakeFiles/implicitMethod.dir/implicitMethod.c.o"

# External object files for target implicitMethod
implicitMethod_EXTERNAL_OBJECTS =

implicitMethod: CMakeFiles/implicitMethod.dir/implicitMethod.c.o
implicitMethod: CMakeFiles/implicitMethod.dir/build.make
implicitMethod: /home/bady/lib/petsc-3.16.6-opt/lib/libpetsc.so
implicitMethod: CMakeFiles/implicitMethod.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bady/HPC_FinalProject/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable implicitMethod"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/implicitMethod.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/implicitMethod.dir/build: implicitMethod

.PHONY : CMakeFiles/implicitMethod.dir/build

CMakeFiles/implicitMethod.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/implicitMethod.dir/cmake_clean.cmake
.PHONY : CMakeFiles/implicitMethod.dir/clean

CMakeFiles/implicitMethod.dir/depend:
	cd /home/bady/HPC_FinalProject/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bady/HPC_FinalProject/src /home/bady/HPC_FinalProject/src /home/bady/HPC_FinalProject/build /home/bady/HPC_FinalProject/build /home/bady/HPC_FinalProject/build/CMakeFiles/implicitMethod.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/implicitMethod.dir/depend

