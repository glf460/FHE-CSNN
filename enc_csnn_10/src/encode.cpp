#include "../include/encode.hpp"
#include <iostream>

/*
对一维向量进行Poisson编码
*/


LweSample * Poisson_encode1dim_enc(LweSample * x, int ndim){

    int n = ndim;
    random_device seed;
	ranlux48 engine(seed());
    uniform_int_distribution<int> distrib(0, 255);

    auto res = new_LweSample_array(ndim, in_out_params);
    Torus32 mu;
    LweSample *rand_enc = new_LweSample_array(1, in_out_params);
    LweSample * enc_one = new_LweSample_array(1, in_out_params);
    lweNoiselessTrivial(enc_one, mu_boot, in_out_params);
    for(int i=0; i<n;i++){
        int rand = distrib(engine);
        mu = modSwitchToTorus32(rand, msg_space);
        lweNoiselessTrivial(rand_enc, mu, in_out_params);
        lweSubTo(x+i, rand_enc, in_out_params);
        tfhe_bootstrap(res+i, secret->cloud.bk, mu_boot, x+i);
        lweAddTo(res+i, enc_one, in_out_params);
    }
    return res;

}


vector<int> Poisson_encode1dim(vector<int> &x){

    int n = x.size();
    random_device seed;
	ranlux48 engine(seed());
    // uniform_int_distribution<int> distrib(0, 255);
    uniform_real_distribution<double> distrib(0.0, 1.0);

    vector<int> res;
    for(int i=0; i<n;i++){
        double rand = distrib(engine)*255;
        // cout<<"随机数是"<<rand;
        if (x[i] >= rand) {
            res.push_back(2);
        }
        else if (x[i] < rand) {
            res.push_back(0);
        }
        else {
            throw "no match";
        }
    }
    return res;

}



/*
code as one hot 
*/
vector<vector<int>> one_hot(int n_size){
    vector<vector<int>> array(n_size);
    for (int i = 0; i < n_size; i++)
    {
        array[i].resize(n_size);
    }

    return array;
}