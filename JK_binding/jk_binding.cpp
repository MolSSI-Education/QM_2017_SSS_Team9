#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<pybind11/numpy.h> //numpy and python  
#include<iostream>
#include<string>
#include<lawrap/blas.h>
#include<cmath>

namespace py = pybind11; //this is what we are using to bind to python


//using lapack to do operations 
//takes a numpy array and returns a numpy array
std::vector<py::array_t<double>> form_jk(py::array_t<double> I_J,
                            py::array_t<double> D_J,
                            py::array_t<double> I_K,
                            py::array_t<double> D_K)
{
    std::vector<py::array_t<double>> list_arr;
    py::buffer_info I_J_info = I_J.request(); //function you call to get buffer info
    py::buffer_info D_J_info = D_J.request(); //function you call to get buffer info
    py::buffer_info I_K_info = I_K.request(); //function you call to get buffer info
    py::buffer_info D_K_info = D_K.request(); //function you call to get buffer info
    
    size_t n_j = D_J_info.shape[0];
    
    const double * I_J_data = static_cast<double *>(I_J_info.ptr); //cast void * to double * 
    const double * D_J_data = static_cast<double *>(D_J_info.ptr); //cast void * to double * 
    const double * I_K_data = static_cast<double *>(I_K_info.ptr); //cast void * to double * 
    const double * D_K_data = static_cast<double *>(D_K_info.ptr); //cast void * to double * 
    
    std::vector<double> J_data(n_j); //std::vector calls new but will automatically delete when out of scope
    std::cout << "before gemv" << '\n';

    LAWrap::gemv('N', n_j, n_j, 1, I_J_data, n_j, D_J_data, 1, 0, J_data.data(), 1);
    std::cout << "after gemv" << '\n';

    py::buffer_info Jbuf = 
    {
        J_data.data(), //pointer 
        //pybind11 takes pointer and shape, copies data, allocates new memory - so even though pointer go out of scope okay
        sizeof(double), //size of double
        py::format_descriptor<double>::format(),//format descriptor
        1, //number of dimensions 
        {n_j},//automatically construct a vector with whatever in curly braces 
        {1 * sizeof(double)} //leading dimension
    };
    //std vector initialization with {}
    //std::vector<double> s = {1.0,2.0,..}

    size_t n = D_K_info.shape[0];
    
    std::vector<double> K_data(n*n); //std::vector calls new but will automatically delete when out of scope

    for(int p = 0; p < n; p++)
    {
        for(int r = 0; r < n; r++)
        {
            for(int q = 0; q < n; q++)
            {
                for(int s = 0; s < n; s++)
                {
                    int test = p*pow(n,3)+r*pow(n,2)+q*n + s;
                    K_data[p*n + q] += I_K_data[test] * D_K_data[r*n + s]; 
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

    list_arr.push_back(Jbuf);       
    list_arr.push_back(Kbuf);       
    return list_arr;
}
PYBIND11_PLUGIN(jk_binding) //need this function to use (macros?) 
//wrap c binding for you 
{
    py::module m("jk_binding", "A basic module");
    
    m.def("form_jk", &form_jk);
    //converts python list to std vector and vice versa
    //need to include pybind11 stl.h 
    return m.ptr();
}

