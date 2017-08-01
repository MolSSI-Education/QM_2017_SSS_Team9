cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(molssi_sss_pybind)

# This must be run using a conda environment
if("$ENV{CONDA_PREFIX}" STREQUAL "")
	message(FATAL_ERROR "This must be run within the MolSSI SSS conda prefix. "
						"Delete the build directory and try again")
endif()

# CMake will sometimes find the conda python version
# (rather the the python inside the sss environment)
set(PYTHON_EXECUTABLE $ENV{CONDA_PREFIX}/bin/python3)

# Find the pybind11 in the conda path
set(PYBIND11_CPP_STANDARD -std=c++11 )
find_package(pybind11 CONFIG REQUIRED
			 PATHS $ENV{CONDA_PREFIX}
			 NO_DEFAULT_PATH)

#find_package(OpenMP)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
add_definitions(-fopenmp)
message(STATUS "Found pybind11: ${pybind11_CONFIG}")

set(CMAKE_INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH True)

# Creates a python module named "module_name"
#name has to match name you give the pybind11 plugin
#in established usually export.cpp
pybind11_add_module(jk_binding MODULE jk_binding.cpp)
#target_link_libraries(jk_binding PRIVATE mkl_intel_lp64 mkl_sequential mkl_core) 
