#include<fstream>
#include<iostream>
#include<lawrap/blas.h>
#include<string>
//print functions
//matrix class
void read_function(double* matrix, std::string filename, int nbas)
{
    std::ifstream input(filename);
    double number;
    for (int i = 0; i < nbas; i++)
    {
        for (int j = 0; j < nbas; j++)
        {
            input >> number;
            matrix[i + j*nbas] = number;
            //std::cout << matrix[i + j*nbas] << '\n';
        }
    }
}

void add_matrices(double* matrix_in1, double* matrix_in2, double* matrix_out, int nbas)
{
    for (int i = 0; i < nbas; i++)
    {
        for (int j = 0; j < nbas; j++)
        {
            matrix_out[i*nbas + j] = matrix_in1[i*nbas +j] + matrix_in2[i*nbas + j];
        }
    }
}
int main()
{
    int nbas = 7;
    int nocc = 5;
    double* H = new double[nbas*nbas];
    double* F = new double[nbas*nbas];
    double* C = new double[nbas*nbas];
    double* S = new double[nbas*nbas];
    read_function(H, "H.data", nbas);
    read_function(F, "F.data", nbas);
    read_function(C, "C.data", nbas);
    read_function(S, "S.data", nbas);
    
    double* D = new double[nbas*nbas];
    LAWrap::gemm('N', 'T', nbas, nbas, nocc, 2, C, nbas, C, nbas, 0, D, nbas);

//    for (int i = 0; i < nbas; i++)
//    {
//        for (int j = 0; j < nbas; j++)
//        {
//            std::cout << D[i*nbas + j] << '\n';
//        }
//    }
    
    double* DS = new double[nbas*nbas];
    LAWrap::gemm('N', 'N', nbas, nbas, nbas, 1, D, nbas, S, nbas, 0, DS, nbas);
   // double nelec = LAWrap::dot(nbas*nbas, D, nbas, S, nbas);
    double nelec = 0;
    for (int i = 0; i < nbas; i++)
    {
        nelec += DS[i*nbas + i];
    }   
    std::cout << nelec << '\n';
    
    double* F_plus_H = new double[nbas*nbas];
    add_matrices(F, H, F_plus_H, nbas);

    
    double scf_energy = LAWrap::dot(nbas*nbas, F_plus_H, 1, D, 1); //works because symmetric matrix
    scf_energy /= 2;

    std::cout << scf_energy << '\n';
    return 0;
}
