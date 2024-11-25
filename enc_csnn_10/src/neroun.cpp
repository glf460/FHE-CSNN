#include "../include/neroun.hpp"
#include "lwesamples.h"
#include "tgsw_functions.h"
#include <cmath>
#include <cstdint>
#include <vector>



Enc_LIFnode::Enc_LIFnode(int ndim, int v_threshold, bool is_LIF, int tau){
    this->V_threshold = v_threshold;
    this->ndim = ndim;

    this->max_value = 0;
    this->decay_tau = tau;

    this->enc_V = new_LweSample_array(ndim, in_out_params);
    this->enc_S = new_LweSample_array(ndim, in_out_params);

    for(int i=0;i<ndim;i++){
        lweNoiselessTrivial(enc_V+i, 0, in_out_params);
    }

    this->is_LIF = is_LIF;

   // this->V_threshold = v_threshold;
    this->enc_V_threshold = new_LweSample_array(1, in_out_params);
    lweNoiselessTrivial(this->enc_V_threshold, modSwitchToTorus32(tau*v_threshold, msg_space), in_out_params);
    //阈值被放大为2倍

    enc_one = new_LweSample_array(1, in_out_params);
    lweNoiselessTrivial(enc_one, mu_boot, in_out_params);

}

void Enc_LIFnode::enc_charge(LweSample * x, int x_dim){
    if (x_dim != this->ndim) {  
        throw "dimention does't match!";
    }

    for(int i=0;i<this->ndim;i++){
        // v = v + x
        lweAddTo(this->enc_V+i, x+i, in_out_params);
        auto score = lweSymDecrypt(this->enc_V+i, secret->lwe_key, msg_space);
        score = modSwitchFromTorus32(score, msg_space);
        score = score<msg_space/2 ? score :score-msg_space;
        if (score > max_value) 
        {
            max_value = score;
        }
    }

}

LweSample * Enc_LIFnode::enc_fire_and_reset(){

   vector<thread> threads; 
    // fire
    for(int i=0; i<this->ndim; i++){
            threads.emplace_back(mem_fn(&Enc_LIFnode::thread_fire_reset),this,i);
        

        // 达到最大线程数时等待其中一个线程完成
        if (threads.size() >= 512)
        {
            threads.front().join();
            threads.erase(threads.begin());
        }
    }

    // 等待剩余线程完成
    for (auto& thread : threads)
    {
        thread.join();
    }

    return this->enc_S;
}

void Enc_LIFnode::thread_fire_reset(int i)
{
    LweSample * temp(enc_V+i);  // 下去测试一下
    lweSubTo(temp, this->enc_V_threshold, in_out_params);
    tfhe_bootstrap_FFT(this->enc_S+i, secret->cloud.bkFFT, mu_boot, temp);
    lweAddTo(this->enc_S+i, enc_one, in_out_params);  // s[i] = s[i] + 1
    if (this->is_LIF) 
    {
        my_LIF_bootstrap(this->enc_V+i, secret->cloud.bkFFT, secret->cloud.bk, this->decay_tau*this->V_threshold, this->enc_V+i, this->decay_tau);
    }
    else 
    {
        my_IF_bootstrap(this->enc_V+i, secret->cloud.bkFFT, secret->cloud.bk, this->decay_tau*this->V_threshold, this->enc_V+i);
    }

}

LweSample * Enc_LIFnode::enc_forward(LweSample * x, int x_dim)
{
    if (x_dim != this->ndim) {
        cout << "x_dim = " << x_dim << ", while this->ndim = " << this->ndim << endl;  
        throw "dimention does't match!";
    } 
    this->enc_charge(x, x_dim);
    this->enc_fire_and_reset();

    for (int i = 0; i<x_dim; i++) 
    {
        auto score = lweSymDecrypt(this->enc_S+i, secret->lwe_key, msg_space);
        lweNoiselessTrivial(this->enc_S+i, score, in_out_params);
    }


    return this->enc_S;
}

void Enc_LIFnode::enc_reset_state(){
    for(int i=0;i<ndim;i++){
        lweNoiselessTrivial(enc_V+i, 0, in_out_params);
    }
}

int Enc_LIFnode :: show_max(){
    return this->max_value;
}







void LIF_FFT(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t volty, const LweSample *x, int tau) {

    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t Nx2 = 2 * N;
    const int32_t n = in_params->n;

    TorusPolynomial *testvect = new_TorusPolynomial(N);
    int32_t *bara = new int32_t[N];

    int32_t barb = modSwitchFromTorus32(x->b, 2*N);
    for (int32_t i = 0; i < n; i++) {
        bara[i] = modSwitchFromTorus32(x->a[i], 2*N);
    }

    //the initial testvec = [0,1,2,...,volty, 0, 0, 0...]
    //这里要变成原来的1/2才能保证电压是一致的。

    for (int32_t i = 0; i<volty*Nx2/msg_space; i++){     //最重要，在填充系数
        auto mu_N = modSwitchToTorus32(int(i*msg_space/Nx2), msg_space);
        // auto mu_N = modSwitchToTorus32(i, Nx2);

        testvect->coefsT[i] = round(double(tau-1)/tau*mu_N);      //mu_N为 明文嵌入密文放大的倍速
        
    } 
    for (int32_t i = volty*Nx2/msg_space; i < N; i++){
            testvect->coefsT[i] = 0;
        
    } 


    // tfhe_blindRotateAndExtract(result, testvect, bk->bk, barb, bara, n, bk_params);

    tfhe_blindRotateAndExtract_FFT(result, testvect, bk->bkFFT, barb, bara, n, bk_params);

    delete[] bara;
    delete_TorusPolynomial(testvect);
}


void IF_FFT(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t volty, const LweSample *x) {

    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t Nx2 = 2 * N;
    const int32_t n = in_params->n;

    TorusPolynomial *testvect = new_TorusPolynomial(N);
    int32_t *bara = new int32_t[N];

    int32_t barb = modSwitchFromTorus32(x->b, 2*N);
    for (int32_t i = 0; i < n; i++) {
        bara[i] = modSwitchFromTorus32(x->a[i], 2*N);
    }

    //the initial testvec = [0,1,2,...,volty, 0, 0, 0...]
    //这里要变成原来的1/2才能保证电压是一致的。

    for (int32_t i = 0; i<volty*Nx2/msg_space; i++){
        auto mu_N = modSwitchToTorus32(int(i*msg_space/Nx2), msg_space);
        // auto mu_N = modSwitchToTorus32(i, Nx2);

        testvect->coefsT[i] = mu_N;
        
    } 
    for (int32_t i = volty*Nx2/msg_space; i < N; i++){
            testvect->coefsT[i] = 0;
        
    } 


    // tfhe_blindRotateAndExtract(result, testvect, bk->bk, barb, bara, n, bk_params);

    tfhe_blindRotateAndExtract_FFT(result, testvect, bk->bkFFT, barb, bara, n, bk_params);

    delete[] bara;
    delete_TorusPolynomial(testvect);
}


void my_tfhe_bootstrap_woKS(LweSample *result, const LweBootstrappingKey *bk, int32_t volty, const LweSample *x) {

    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t Nx2 = 2 * N;
    const int32_t n = in_params->n;

    TorusPolynomial *testvect = new_TorusPolynomial(N);
    int32_t *bara = new int32_t[N];

    int32_t barb = modSwitchFromTorus32(x->b, 2*N);
    for (int32_t i = 0; i < n; i++) {
        bara[i] = modSwitchFromTorus32(x->a[i], 2*N);
    }

    //the initial testvec = [0,1,2,...,volty, 0, 0, 0...]

    for (int32_t i = 0; i<volty*Nx2/msg_space; i++){
        auto mu_N = modSwitchToTorus32(int(i*msg_space/Nx2), msg_space);
        // auto mu_N = modSwitchToTorus32(i, Nx2);

        testvect->coefsT[i] = mu_N;
        
    } 
    for (int32_t i = volty*Nx2/msg_space; i < N; i++){
            testvect->coefsT[i] = 0;
        
    } 


    tfhe_blindRotateAndExtract(result, testvect, bk->bk, barb, bara, n, bk_params);


    delete[] bara;
    delete_TorusPolynomial(testvect);
}


void my_IF_bootstrap(LweSample *result, const LweBootstrappingKeyFFT *bkFFT, const LweBootstrappingKey *bk, int32_t volty, const LweSample *x) {

    LweSample *u = new_LweSample(&bkFFT->accum_params->extracted_lweparams);

    IF_FFT(u, bkFFT, volty, x);
    // my_tfhe_bootstrap_woKS(u, bk, volty, x);

    // Key Switching
    lweKeySwitch(result, bkFFT->ks, u);

    delete_LweSample(u);
}

void my_LIF_bootstrap(LweSample *result, const LweBootstrappingKeyFFT *bkFFT, const LweBootstrappingKey *bk, int32_t volty, const LweSample *x, int tau) {

    LweSample *u = new_LweSample(&bkFFT->accum_params->extracted_lweparams);

    LIF_FFT(u, bkFFT, volty, x, tau);
    // my_tfhe_bootstrap_woKS(u, bk, volty, x);

    // Key Switching
    lweKeySwitch(result, bkFFT->ks, u);

    delete_LweSample(u);
}



