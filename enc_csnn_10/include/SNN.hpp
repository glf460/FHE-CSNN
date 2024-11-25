#pragma once
#include "Layer.hpp"
#include "neroun.hpp"
#include <stdexcept>
#include <limits>
#include <string>
#include <vector>
#include "tfhe_core.h"
#include "tfhe_import.hpp"

using namespace std;

class Enc_Snn{
private:
    Enc_Linear fc1;
    Enc_LIFnode ac1;
    int scale_down;

public:
    Enc_Snn(int);


    LweSample * enc_forward(LweSample *, int);

    int show_scale_down();

    void resrt_volty();

    void save(string posi_weights);

    void load(string posi_weights);
};


class Enc_SNNDF{
private:
    Enc_Linear fc1;
    Enc_LIFnode ac1;
    Enc_Linear fc2;
    Enc_LIFnode ac2;

    const int layers = 2;
    int scale_down;

public:
    Enc_SNNDF(int);


    LweSample * enc_forward(LweSample *, int);

    int show_scale_down();

    void resrt_volty();

    void save(int, vector<string>);

    void load(int, vector<string>s);
};

class Enc_CSNN
{
    private:
        Conv2D conv;
        Enc_LIFnode lif1;
        Avgpooling2D mp;

        Enc_Linear fc;
        Enc_LIFnode lif2;
        Enc_Linear fc2;
        Enc_LIFnode lif3;

        int scale;
        int hidden_dim;

        int decay_tau;

    public:
        //  net( scale_down,     10,            1,          3,            1,         1,          2,              160,         false,    TAU);
        Enc_CSNN(int scale ,int kernel_nums, int Vth, int kernelsize, int stride, int pad, int poolingsize, int hidden_dim, bool LIF, int tau);

        bool load(vector<string> path, bool decayinput);

        bool save(string path);

        LweSample* enc_forward(LweSample* input, int channel, int length, int width);

        void reset_volty();

        void show_max_value();

};
