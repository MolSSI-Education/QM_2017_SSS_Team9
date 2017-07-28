#include<fstream>
#include<iostream>
#include<lawrap/blas.h>
#include<string>
#include<vector>
#include<algorithm>

void read_function(std::vector<double> & vec, std::string filename, int nbas) // pass the vector by reference to avoid embarassment
{																			  // its not advisable to use shared_ptr<std::vector<double>>	
    std::ifstream input(filename);											  // as it adds an extra layer of pointers to the vector object  
    double number;													          // which already uses shared_ptrs under the hood. So, shared_ptrs	
    for (int i = 0; i < nbas * nbas; i++){									  // are very useful for custom classes. Fot standard container objects 	
            input >> number;											      // unique_ptrs should be preferred over shared_ptrs.
            vec.push_back(number); // row major format
        }
}
int main()
{
    int nbas = 7;
    int nocc = 5;
	std::vector<double> H;
	std::vector<double> F;
	std::vector<double> C;
	std::vector<double> S;
	std::vector<double> D (nbas * nbas);
	std::vector<double> DS (nbas * nbas);
	std::vector<double> F_plus_H (nbas * nbas);  // allocate memory before use 

    read_function(H, "H.data", nbas);
    read_function(F, "F.data", nbas);
    read_function(C, "C.data", nbas);
    read_function(S, "S.data", nbas);
    
    LAWrap::gemm('T', 'N', nbas, nbas, nocc, 2, C.data(), nbas, C.data(), nbas, 0, D.data(), nbas); // everything in column major format: D = 2.0 * C.T * C

    LAWrap::gemm('N', 'T', nbas, nbas, nbas, 1, D.data(), nbas, S.data(), nbas, 0, DS.data(), nbas);

    double nelec = 0;
    for (int i = 0; i < nbas; i++)
    {
        nelec += DS[i*nbas + i];
    }   
    std::cout << nelec << '\n';

    //F_plus_H = H;   // '='  operator overloading
    
    std::transform(H.begin(), H.end(), F.begin(), 
               F_plus_H.begin(), std::plus<double>());  // one can write own '+' operator overloading


    double scf_energy = LAWrap::dot(nbas*nbas, F_plus_H.data(), 1, D.data(), 1); 
    scf_energy /= 2;

    std::cout << scf_energy << '\n';

    return 0;
}
