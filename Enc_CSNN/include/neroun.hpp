#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <limits>
#include <thread>
#include <functional>
#include "tfhe_core.h"
#include "tfhe_import.hpp"



using namespace std;


class Enc_LIFnode{

    private:
        int max_value;  //初始值为0，  可能用于记录最大值

        int ndim;
        string reset_mod;
        LweSample * enc_one;  //加密的1  可能是vreset
        
        //! store the intermediate process
        bool is_LIF;
        vector<vector<double>> stored_V_seq;
        vector<vector<double>> stored_S_seq;

        //! current volty
        LweSample* enc_V;  //存储现在的电压 初始值为0  
        LweSample* enc_S;  

        //! The threshold for firing
        int V_threshold;  

        LweSample * enc_V_threshold;  //阈值电压，初始为decay_tau*v_threshold

        int decay_tau;


    public:

        Enc_LIFnode(int ndim, int v_threshold, bool is_LIF = true, int tau = 2);

        void enc_charge(LweSample *, int); //充电 更新电压enc_V  更新max_value

        LweSample * enc_fire_and_reset();

        LweSample * enc_forward(LweSample *, int);

        void enc_reset_state();

        void thread_fire_reset(int i);   //第i个神经元的fire_reset

        int show_max();


        bool save(string filename);
        bool load(string filename);

};


void my_IF_bootstrap(LweSample *result, const LweBootstrappingKeyFFT *bkFFT, const LweBootstrappingKey *bk, int volty, const LweSample *x);
void my_LIF_bootstrap(LweSample *result, const LweBootstrappingKeyFFT *bkFFT, const LweBootstrappingKey *bk, int volty, const LweSample *x, int tau);

