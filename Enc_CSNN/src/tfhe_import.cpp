#include "../include/tfhe_import.hpp" // 导入自定义的TFHE库
#include <iostream>  // 导入标准的输入输出流库

// 声明一个函数our_default_gate_bootstrapping_parameters，该函数返回一个TFheGateBootstrappingParameterSet指针，并接收一个整数参数minimum_lambda
TFheGateBootstrappingParameterSet *our_default_gate_bootstrapping_parameters(int minimum_lambda);

// Security。定义了一些与安全性相关的常量
// 定义一个名为 minimum_lambda 的常量整数，并将其值设定为 SECLEVEL。SECLEVEL 是一个预定义的宏（或常量），用于指定加密系统的最低安全级别（通常是安全参数）
const int minimum_lambda = SECLEVEL;
// 定义一个名为 noisyLWE 的常量布尔值，并将其值设定为 SECNOISE。SECNOISE 是一个预定义的宏（或常量），用于指示加密系统是否使用噪声 LWE（Learning With Errors）方案
const bool noisyLWE      = SECNOISE;
// 定义一个名为 alpha 的常量双精度浮点数，并将其值设定为 SECALPHA。SECALPHA 是一个预定义的宏（或常量），用于指定加密系统中的噪声参数 alpha
const double alpha       = SECALPHA;

// 定义一个名为 msg_space 的整数变量，并将其值设定为 MSG_SIZE。MSG_SIZE 是一个预定义的宏（或常量），用于指定消息空间的大小
int msg_space = MSG_SIZE;


// 定义了一个外部常量mu_boot，它通过函数modSwitchToTorus32将值1转换为Torus32格式，消息空间大小为msg_space
// Program the wheel to value(s) after Bootstrapping
extern const Torus32 mu_boot = modSwitchToTorus32(1, msg_space);

// 调用之前声明的函数our_default_gate_bootstrapping_parameters，生成TFHE参数集并存储在指针params中
TFheGateBootstrappingParameterSet *params = our_default_gate_bootstrapping_parameters(minimum_lambda);
// TFheGateBootstrappingParameterSet *params = new_default_gate_bootstrapping_parameters(minimum_lambda);
// 从params中获取in_out_params参数
const LweParams *in_out_params   = params->in_out_params;

// 生成一个新的随机引导加密密钥集secret，并从中提取快速傅里叶变换（FFT）形式的引导加密密钥bs_key
TFheGateBootstrappingSecretKeySet *secret = new_random_gate_bootstrapping_secret_keyset(params);
const LweBootstrappingKeyFFT *bs_key = secret->cloud.bkFFT;

// 定义一个名为 our_default_gate_bootstrapping_parameters 的函数，返回一个 TFheGateBootstrappingParameterSet 指针，并接收一个整数参数 minimum_lambda
// 实现函数our_default_gate_bootstrapping_parameters，当minimum_lambda大于128时，输出错误信息，因为当前只实现了约128位安全性的参数
TFheGateBootstrappingParameterSet *our_default_gate_bootstrapping_parameters(int minimum_lambda)
{
    // 如果 minimum_lambda 大于128，输出错误信息，因为当前只实现了约128位安全性的参数
    if (minimum_lambda > 128)
        cerr << "Sorry, for now, the parameters are only implemented for about 128bit of security!\n";

    // 下面定义了许多与TFHE参数相关的静态常量，这些常量从预定义的宏中获取
    static const int n = SEC_PARAMS_n; // n: LWE参数中的维度
    static const int N = SEC_PARAMS_N; // N: TLWE参数中的多项式环的维度
    static const int k = SEC_PARAMS_k; // k: TLWE参数中的Torus多项式数量
    static const double max_stdev = SEC_PARAMS_STDDEV; // max_stdev: 最大标准偏差

    static const int bk_Bgbit = SEC_PARAMS_BK_BASEBITS; //<-- ld, thus: 2^10  bk_Bgbit: 基本bit位数
    static const int bk_l = SEC_PARAMS_BK_LENGTH;       // bk_l: 基长度
    static const double bk_stdev = SEC_PARAMS_BK_STDDEV; // bk_stdev: 自举密钥的标准偏差

    static const int ks_basebit = SEC_PARAMS_KS_BASEBITS; //<-- ld, thus: 2^1   ks_basebit: 密钥切换基的bit位数
    static const int ks_length = SEC_PARAMS_KS_LENGTH;    // ks_length: 密钥切换基的长度
    static const double ks_stdev = SEC_PARAMS_KS_STDDEV;  // ks_stdev: 密钥切换的标准偏差

    // 创建了LWE、TLWE和TGSW参数对象，分别使用之前定义的常量
    LweParams  *params_in    = new_LweParams (n,    ks_stdev, max_stdev);  // params_in: LWE参数对象，使用n、ks_stdev和max_stdev
    TLweParams *params_accum = new_TLweParams(N, k, bk_stdev, max_stdev);  // params_accum: TLWE参数对象，使用N、k、bk_stdev和max_stdev
    TGswParams *params_bk    = new_TGswParams(bk_l, bk_Bgbit, params_accum);  // params_bk: TGSW参数对象，使用bk_l、bk_Bgbit和params_accum

    // 将这些参数对象注册到TFHE垃圾回收器中，以便自动管理内存，防止内存泄漏
    TfheGarbageCollector::register_param(params_in); 
    TfheGarbageCollector::register_param(params_accum); 
    TfheGarbageCollector::register_param(params_bk);

    // 创建并返回一个新的TFheGateBootstrappingParameterSet对象，该对象包含LWE和TGSW参数
    // 该对象包含：ks_length: 密钥切换基的长度、ks_basebit: 密钥切换基的bit位数、params_in: LWE参数、params_bk: TGSW参数
    return new TFheGateBootstrappingParameterSet(ks_length, ks_basebit, params_in, params_bk);
}



