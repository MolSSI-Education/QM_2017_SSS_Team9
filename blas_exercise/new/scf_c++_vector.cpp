#include <fstream>
#include <iostream>
#include <lawrap/blas.h>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>

/*std::vector<double> operator+(std::vector<double>& left_vec, std::vector<double>& right_vec)
{ 
    if (left_vec.size() > right_vec.size())
    {
        std::vector<double> result = left_vec;
        for(int i = 0; i < right_vec.size(); i++)
       {
            result[i] += right_vec[i];
       }
       return result;
    }
    else 
    {
      std::vector<double> result = right_vec;
      for (int i =0; i < left_vec.size(); i++)
      {
          result[i] += left_vec[i];
      }
      return result;
    } 
    
}


  


void read_function(std::vector<double> & vec, std::string filename, int nbas) 
{																			 
    std::ifstream input(filename);											
    double number;													       
    for (int i = 0; i < nbas * nbas; i++){								
            input >> number;											      
            vec.push_back(number); // row major format
        }
}

// pass the vector by reference to avoid embarassment
// its not advisable to use shared_ptr<std::vector<double>>   
// as it adds an extra layer of pointers to the vector object  
// which already uses shared_ptrs under the hood. So, shared_ptrs 
// are very useful for custom classes. Fot standard container objects     
// unique_ptrs should be preferred over shared_ptrs.



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
    F_plus_H = H + F;       // implemented '+' operator overloading. The following is the transform function which was used before
   // std::transform(H.begin(), H.end(), F.begin(), 
     //          F_plus_H.begin(), std::plus<double>());


    double scf_energy = LAWrap::dot(nbas*nbas, F_plus_H.data(), 1, D.data(), 1); 
    scf_energy /= 2;

    std::cout << scf_energy << '\n';

    return 0;
}
*/

void read_function(std::shared_ptr<std::vector<double>>  vec, std::string filename, int nbas)
{
    std::ifstream input(filename);
    double number; 
    for (int i = 0; i < nbas * nbas; i++){
            input >> number;
            vec->push_back(number); // row major format
        }
}



int main()
{
    int nbas = 7;
    int nocc = 5;
	auto H = std::make_shared<std::vector<double>>();
	auto F = std::make_shared<std::vector<double>>();
	auto C = std::make_shared<std::vector<double>>();
	auto S = std::make_shared<std::vector<double>>();

    read_function(H, "H.data", nbas);
    read_function(F, "F.data", nbas);
    read_function(C, "C.data", nbas);
    read_function(S, "S.data", nbas);

	auto D = std::make_shared<std::vector<double>>(nbas * nbas);
	auto DS = std::make_shared<std::vector<double>>(nbas * nbas);
	auto F_plus_H = std::make_shared<std::vector<double>>(nbas * nbas);

    LAWrap::gemm('T', 'N', nbas, nbas, nocc, 2, C->data(), nbas, C->data(), nbas, 0, D->data(), nbas); // everything in column major format: D = 2.0 * C.T * C

    LAWrap::gemm('N', 'T', nbas, nbas, nbas, 1, D->data(), nbas, S->data(), nbas, 0, DS->data(), nbas);

    double nelec = 0;
    for (int i = 0; i < nbas; i++)
    {
        nelec += (* DS)[i*nbas + i];
    }
    std::cout << nelec << '\n';

    F_plus_H = H;   // '='  operator overloading
    std::transform(H->begin(), H->end(), F->begin(), 
              F_plus_H->begin(), std::plus<double>());


    double scf_energy = LAWrap::dot(nbas*nbas, F_plus_H->data(), 1, D->data(), 1);
    scf_energy /= 2;

    std::cout << scf_energy << '\n';

    return 0;
}

