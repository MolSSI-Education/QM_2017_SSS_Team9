#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<pybind11/numpy.h> //numpy and python  
#include<iostream>
#include<string>
#include<lawrap/blas.h>
#include<cmath>
#include<omp.h>

namespace py = pybind11; //this is what we are using to bind to python

//using lapack to do operations 
//takes a numpy array and returns a numpy array
py::array_t<double> form_j(py::array_t<double> I,
                            py::array_t<double> D)
{
    py::buffer_info I_info = I.request(); //function you call to get buffer info
    py::buffer_info D_info = D.request(); //function you call to get buffer info
    
    size_t n = D_info.shape[0];
    
    const double * I_data = static_cast<double *>(I_info.ptr); //cast void * to double * 
    const double * D_data = static_cast<double *>(D_info.ptr); //cast void * to double * 
    
    std::vector<double> J_data(n); //std::vector calls new but will automatically delete when out of scope
    std::cout << "before gemv" << '\n';

    LAWrap::gemv('N', n, n, 1, I_data, n, D_data, 1, 0, J_data.data(), 1);
    std::cout << "after gemv" << '\n';

    py::buffer_info Jbuf = 
    {
        J_data.data(), //pointer 
        //pybind11 takes pointer and shape, copies data, allocates new memory - so even though pointer go out of scope okay
        sizeof(double), //size of double
        py::format_descriptor<double>::format(),//format descriptor
        1, //number of dimensions 
        {n},//automatically construct a vector with whatever in curly braces 
        {1 * sizeof(double)} //leading dimension
    };
    //std vector initialization with {}
    //std::vector<double> s = {1.0,2.0,..}

    return py::array_t<double>(Jbuf);
}
py::array_t<double> form_j_loop(py::array_t<double> I,
                            py::array_t<double> D)
{
    py::buffer_info I_info = I.request(); //function you call to get buffer info
    py::buffer_info D_info = D.request(); //function you call to get buffer info
    
    size_t n = D_info.shape[0];
    
    const double * I_data = static_cast<double *>(I_info.ptr); //cast void * to double * 
    const double * D_data = static_cast<double *>(D_info.ptr); //cast void * to double * 
    
    std::vector<double> J_data(n*n); //std::vector calls new but will automatically delete when out of scope

#pragma omp parallel
{
    std::cout << "number of threads using:" << omp_get_max_threads() << "\n";
#pragma omp for 
        for(int p = 0; p < n; p++)
        {
            for(int q = 0; q <= p; q++)
            {
                for(int r = 0; r < n; r++)
                {
                    for(int s = 0; s <= r; s++)
                    {
                        int test = p*pow(n,3)+q*pow(n,2)+r*n + s;
                        if (r==s)
                            J_data[p*n + q] += I_data[test] * D_data[r*n + s];
                        else 
                            J_data[p*n + q] += 2*(I_data[test] * D_data[r*n + s]);

                        J_data[q*n +p] = J_data[p*n +q];
                    }
                }
            }
        }
}
    py::buffer_info Jbuf = 
    {
        J_data.data(), //pointer 
        //pybind11 takes pointer and shape, copies data, allocates new memory - so even though pointer go out of scope okay
        sizeof(double), //size of double
        py::format_descriptor<double>::format(),//format descriptor
        2, //number of dimensions 
        {n,n},//automatically construct a vector with whatever in curly braces 
        {n * sizeof(double), sizeof(double)} //leading dimension
    };
    //std vector initialization with {}
    //std::vector<double> s = {1.0,2.0,..}

    return py::array_t<double>(Jbuf);
}

py::array_t<double> form_k(py::array_t<double> I,
                            py::array_t<double> D)
{
    py::buffer_info I_info = I.request(); //function you call to get buffer info
    py::buffer_info D_info = D.request(); //function you call to get buffer info
    
    size_t n = D_info.shape[0];
    
    const double * I_data = static_cast<double *>(I_info.ptr); //cast void * to double * 
    const double * D_data = static_cast<double *>(D_info.ptr); //cast void * to double * 
    
    std::vector<double> K_data(n*n); //std::vector calls new but will automatically delete when out of scope

#pragma omp parallel for
    for(int p = 0; p < n; p++)
    {
        for(int r = 0; r < n; r++)
        {
            for(int q = 0; q <= p; q++)
            {
                for(int s = 0; s < n; s++)
                {
                    int test = p*pow(n,3)+r*pow(n,2)+q*n + s;
                    K_data[p*n + q] += I_data[test] * D_data[r*n + s];
                    K_data[q*n +p] = K_data[p*n +q];
                }
            }
        }
    }

    py::buffer_info Kbuf = 
    {
        K_data.data(), //pointer 
        //pybind11 takes pointer and shape, copies data, allocates new memory - so even though pointer go out of scope okay
        sizeof(double), //size of double
        py::format_descriptor<double>::format(),//format descriptor
        2, //number of dimensions 
        {n,n},//automatically construct a vector with whatever in curly braces 
        {n * sizeof(double), sizeof(double)} //leading dimension
    };
    //std vector initialization with {}
    //std::vector<double> s = {1.0,2.0,..}

    return py::array_t<double>(Kbuf);
}
PYBIND11_PLUGIN(jk_binding) //need this function to use (macros?) 
//wrap c binding for you 
{
    py::module m("jk_binding", "A basic module");
    
    m.def("form_j", &form_j);
    m.def("form_k", &form_k);
    m.def("form_j_loop", &form_j_loop);
    //converts python list to std vector and vice versa
    //need to include pybind11 stl.h 
    return m.ptr();
}

