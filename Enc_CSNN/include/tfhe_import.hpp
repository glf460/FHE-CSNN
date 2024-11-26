#include "../tfhe/include/tfhe.h"
#include "../tfhe/include/numeric_functions.h"
#include "../tfhe/include/tfhe_core.h"
#include "../tfhe/include/tfhe_garbage_collector.h"
#include "../tfhe/include/tgsw_functions.h"
#include "../tfhe/include/tlwe.h"
#include "../tfhe/include/tgsw.h"
#include "../tfhe/include/lwesamples.h"
#include "../tfhe/include/lwekey.h"
#include "../tfhe/include/lweparams.h"
#include "../tfhe/include/polynomials.h"
#include <iostream>


using namespace std;

// Defines
#define VERBOSE 1


// Security constants
#define SECLEVEL 10
#define SECNOISE true
#define SECALPHA pow(2., -20)
#define SEC_PARAMS_STDDEV    pow(2., -30)
#define SEC_PARAMS_n  128                   //  LweParams
#define SEC_PARAMS_N  8192*4               // TLweParams
#define MSG_SIZE 4096*4
#define SEC_PARAMS_k    1                   // TLweParams
#define SEC_PARAMS_BK_STDDEV pow(2., -36)   // TLweParams
#define SEC_PARAMS_BK_BASEBITS 10           // TGswParams
#define SEC_PARAMS_BK_LENGTH    3           // TGswParams
#define SEC_PARAMS_KS_STDDEV pow(2., -25)   // Key Switching Params
#define SEC_PARAMS_KS_BASEBITS  1           // Key Switching Params
#define SEC_PARAMS_KS_LENGTH   18           // Key Switching Params

extern const Torus32 mu_boot;
extern TFheGateBootstrappingParameterSet *params;
extern const LweParams *in_out_params;
extern TFheGateBootstrappingSecretKeySet *secret;
extern const LweBootstrappingKeyFFT *bs_key;

extern int msg_space;

extern const double alpha;






