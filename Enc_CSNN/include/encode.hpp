#include <vector>
#include <random>
#include <iostream>
#include "tfhe_import.hpp"

using namespace std;
/*
poisoon encode                                                                                                                                                                                              `
*/
LweSample * Poisson_encode1dim_enc(LweSample*, int);

/*
对一维向量进行Poisson编码
*/
vector<int> Poisson_encode1dim(vector<int> &);


/*
code as one hot 
*/
vector<vector<int>> one_hot(int);
