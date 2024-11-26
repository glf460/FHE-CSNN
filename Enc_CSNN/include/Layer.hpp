#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <limits>
#include <fstream>
#include "tfhe_core.h"
#include "tfhe_import.hpp"

using namespace std;

class Enc_Linear{
    private:
        bool is_train;
        short n_pre;  // 输入的维度
        short n_post;  // 输出的维度

        vector<vector<int>> weights_matrix;

        LweSample * enc_output;


    public:

        Enc_Linear(short n_pre, short n_post, bool is_train=false);

        LweSample * enc_forward(LweSample *, int); //线性层前向传播，输入为上一层的输出和长度 

        void update(){};

        bool save(string);  //保存权重到文件里  输入文件路径

        bool load(string, int scale); //从文件里加载权重，存到weights_matrix 输入文件路径和缩放比例

};


class Avgpooling2D{
    private:
        // 这一层的固有属性
        bool is_padding;
        const int kernel_size; // 池化的局部窗口直径
        const int step;        // 池化的步长, 一般和 kernel_size 相等
        // 缓冲区, 避免每次重新分配的
        LweSample* output;  // 记录上一次输出

        const int input_length;
        const int input_width;
        const int input_channels;

    public:
        Avgpooling2D(int size, int stride, bool padding, int channel, int length, int width);

        LweSample * enc_forward(LweSample *, int channel, int length, int width);

        int posi_index(int channel, int x, int y);

        void avg_pooling(LweSample *, int channel, int x, int y, LweSample *);

        int out_dimension();

}; 


class Conv2D{
    private:
        // 这一层的固有属性
        const int padding; //padding的数量
        const int kernel_size; // 卷积的局部窗口直径
        const int step;        // 卷积的步长, 一般和 kernel_size 相等
        const int out_channels;
        int out_H; 
        int out_W; 
        
        // 缓冲区, 避免每次重新分配的
        LweSample* output;  // 记录上一次输出

        const int in_length;
        const int in_width;
        const int in_channels;

        // kernel卷积参数
        vector<vector<double>> kernels;

    public:
        Conv2D(int kernel_size, int out_channels, int stirde, int padding, int in_ch, int in_le, int in_width);

        LweSample* forward(LweSample* input, int channel, int length, int width);

        void compute_conv(LweSample* output, LweSample* input, int out_channel, int x, int y);

        int kernel_index(int channel, int x, int y);

        int img_index(int channel, int x, int y);

        bool load(string, int);

        bool save(string);

        int out_dimension();

        int outH();

        int outW();

        int outchannel();
};


// 用于分割字符串 输入为字符串和分隔符 输出为分割后的字符串数组 
vector<string> split(const string &str, const string &pattern);
