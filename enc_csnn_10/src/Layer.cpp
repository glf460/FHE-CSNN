#include "../include/Layer.hpp"
#include "lweparams.h"
#include "lwesamples.h"
#include "tfhe_core.h"
#include <csignal>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <ios>
#include <math.h>
#include <string>
#include <utility>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <iomanip>


Enc_Linear::Enc_Linear(short n_pre, short n_post, bool is_train){
    this->is_train = is_train;
    this->n_post = n_post;
    this->n_pre = n_pre;
    this->weights_matrix = vector<vector<int>>(n_post, vector<int> (n_pre, 1));
 
    this->enc_output = new_LweSample_array(n_post, in_out_params);
}

bool Enc_Linear:: save(string posi_weight){
    ofstream file;
    file.open(posi_weight, ios_base::out);
    if(!file.is_open()){
        cout << "open file failed!" << endl;
        return 1;
    }
    for(int i=0; i < this->n_pre; i++){
        string line = "";
        for(int j=0; j<this->n_post; j++){
            line += to_string(this->weights_matrix[j][i]);
            line += " ";
        }
        file << line <<endl;
    }
    file.close();
    return 0;

}

bool Enc_Linear:: load(string posi_weight, int scale){
    ifstream file;
    cout << "posi_weight = " << posi_weight << endl;
    file.open(posi_weight, ios_base::in);
    string line;
    int i = 0;
    int max = -100;
    int min = 100;
    while (getline(file, line)) {
        auto split_line = split(line, " ");
        // cout << "split_line["<< i << "]" << split_line[0] << endl;
        // cout << split_line.size() << endl;
        for(int j=0;j<split_line.size();j++){
            // cout << stof(split_line[j]) << endl;
            this->weights_matrix[j][i] = round(stof(split_line[j])*scale);
            if(this->weights_matrix[j][i] > max) max = this->weights_matrix[j][i];
            if (this->weights_matrix[j][i] < min) min = this->weights_matrix[j][i];
        }
        i++;
    }
    cout<<"the max and min weights are"<<max<<"  "<<min<<endl; 

    return 0;
}


LweSample* Enc_Linear :: enc_forward(LweSample *x, int ndim){
    if ( ndim != this->n_pre) {
        throw "dimension doesn't match!";
    }

    for(int i=0; i < this->n_post; i++){
        lweNoiselessTrivial(this->enc_output+i, 0, in_out_params); //初始化输出向量的第 i 个元素为 0。

        for(int j=0; j<this->n_pre; j++){
            //将 weights_matrix[i][j] 与 x[j] 相乘的结果累加到 enc_output[i] 上
            lweAddMulTo(this->enc_output+i, this->weights_matrix[i][j], x+j, in_out_params);
            // output[i] += this->weights_matrix[i][j]*x[j];
        }
    }

    return this->enc_output;
}


Avgpooling2D :: Avgpooling2D(int size, int stride, bool padding,  int channel, int length, int width) :
kernel_size(size),
step(stride),
input_length(length),
input_width(width),
input_channels(channel)
{
    this->is_padding = padding;
    const int out_H = std::floor(((input_length - kernel_size) / step)) + 1;
    const int out_W = std::floor(((input_width - kernel_size) / step)) + 1;

    this->output = new_LweSample_array(input_channels * out_H * out_W, in_out_params);


}

LweSample* Avgpooling2D :: enc_forward(LweSample* input, int channel, int length, int width){
    if ( channel != this->input_channels || length != input_length || width != input_width) {
        throw "dimension doesn't match!";
    }

    int out_index = 0;
    for(int i = 0;i < input_channels; ++i){  // 每个通道
        for(int y = 0; y < input_width-kernel_size + 1 ; y+=step)
        {
            for (int x = 0; x < input_width-kernel_size +1; x+=step) 
            {
                lweNoiselessTrivial(this->output+out_index, 0, in_out_params);
                avg_pooling((this->output + (out_index++)), i, x, y, input);
            }
        }
    }

    return this->output;
}

int Avgpooling2D :: posi_index(int channel, int x, int y){
    int posi = channel*this->input_length*this->input_width + y * this->input_length + x;
    if(posi < input_channels*input_length*input_width){
        return posi;
    }
    else {
        throw "out of index";
    }

}


void Avgpooling2D :: avg_pooling(LweSample* output, int channel, int x, int y, LweSample * input){
    if ( channel >= this->input_channels || x >= input_length || y >= input_width) {
        throw "dimension doesn't match!";
    }    

    for(int i = 0; i<this->kernel_size; i++)
    {
        for (int j = 0; j<this->kernel_size; j++) 
        {
            lweAddTo(output , input+posi_index(channel, x+i, y+j), in_out_params);
        }
    }

}

int Avgpooling2D :: out_dimension()
{
    const int out_H = std::floor(((input_length - kernel_size) / step)) + 1;
    const int out_W = std::floor(((input_width - kernel_size) / step)) + 1;
    return input_channels*out_H*out_W;
}


Conv2D :: Conv2D(int kernel_size, int out_channels, int stirde, int padding, int in_ch, int in_le, int in_width):
padding(padding), //padding的数量
kernel_size(kernel_size), // 卷积的局部窗口直径
step(stirde),        // 卷积的步长(), 一般和 kernel_size 相等
out_channels(out_channels),

in_length(in_le),
in_width(in_width),
in_channels(in_ch)
{
    //二维向量，其中每个内向量代表一个卷积核  大小为 in_ch * kernel_size * kernel_size，初始值为 1
    this->kernels = vector<vector<double>> (out_channels, vector<double> (in_ch * kernel_size*kernel_size, 1));

    this->out_H = std::floor(((in_le - kernel_size + 2 * padding) / step)) + 1;
    this->out_W = std::floor(((in_width - kernel_size + 2 * padding) / step)) + 1;

    // this->output = vector<double>(out_channels * out_H * out_W, 0);
    this->output = new_LweSample_array(out_channels * out_H * out_W, in_out_params);

}

bool Conv2D::load(string path, int scale) {
    ifstream file;
    file.open(path, ios_base::in);
    string line;
    int max = -100;
    int min = 100;
    int i = 0;
    bool is_first_line = true; // 标记是否是第一行

    while(getline(file, line)) {

        auto split_line = split(line, " ");

        cout << "size: " << split_line.size() << endl;
        cout << "kernel size: " << kernel_size << endl;

        // 打印当前行分割后的大小和核大小（仅在第一行打印）
        if (is_first_line)
        {
            // 打印第一行的所有值
            cout << "Values in the first line: ";
            for (const auto &value : split_line)
            {
                cout << value << " "; // 逐个打印值
            }
            cout << endl;          // 第一行打印后换行
            is_first_line = false; // 标记已处理过第一行
            cout << "========================" << endl;
        }

        // 检查分割后的数值数量是否与 kernel_size * kernel_size 相等
        // 检查分割后的数值数量是否与 kernel_size * kernel_size * in_channels 相等
        if (split_line.size() != kernel_size * kernel_size * in_channels) {
            cout << "split_line.size() = "<< split_line.size() << endl;
            cout << "kernel_size = "<< kernel_size << endl;
            throw std::runtime_error("dimension doesn't match!");
        }
        for(int j = 0; j < split_line.size(); j++){
            // cout << stof(split_line[j]) << endl;
            this -> kernels[i][j] = round(stof(split_line[j]) * scale);
            if (this -> kernels[i][j] > max) max = this -> kernels[i][j];
            if (this -> kernels[i][j] < min) min = this -> kernels[i][j];
        }
        i++;

        cout << "conv's weights are " << max << " and "<< min << endl;
    }

    return 0;
}


bool Conv2D :: save(string path){
    ofstream file;
    file.open(path, ios_base::out);
    for(int i=0; i < this->out_channels; i++){
        string line = "";
        for(int j=0; j< kernel_size*kernel_size ; j++){
            line += to_string(this->kernels[i][j]);
            line += " ";
        }
        file << line <<endl;
    }
    return 0;
}


int Conv2D::img_index(int channel, int x, int y)
{
    int posi = channel*this->in_length*this->in_width + y * this->in_length + x;
    if(posi < in_channels*in_length*in_width){
        return posi;
    }
    else {
        throw "out of index";
    }
}

LweSample* Conv2D::forward(LweSample* input, int channel, int length, int width)
{
    if ( channel != in_channels || length != in_length || width != in_width) {
        throw "dimension doesn't match!";
    }

    int out_index = 0;

    for (int oc = 0; oc < out_channels; oc++) 
    {
        for(int y = -padding; y < in_width - kernel_size + padding + 1; y += step)
        {
            for (int x = -padding; x < in_length - kernel_size + padding + 1; x += step) 
            {
                lweNoiselessTrivial(this->output+out_index, 0, in_out_params);
                compute_conv(this->output + (out_index++), input, oc, x, y);
                // this->output[out_index++] = compute_conv(input, oc, x, y);
            }
        }
    }
    return this->output;
}


void Conv2D::compute_conv(LweSample* output, LweSample* input, int out_channel, int x, int y)
{
    for (int c = 0; c < in_channels; c++) 
    {
        for (int i = 0; i<kernel_size; i++) 
        {
            for (int j = 0; j<kernel_size; j++) 
            {
                if(x+i<0 || y+j<0 || x+i >= in_length || y+j >= in_width)
                {
                    continue;
                }
                lweAddMulTo(output,  this->kernels[out_channel][kernel_index(c, i, j)], input+img_index(c, x+i, y+j), in_out_params);

                // res += input[img_index(c, x+i, y+j)] * this->kernels[out_channel][kernel_index(c, i, j)];
            }
        }
    }

}


int Conv2D::kernel_index(int channel, int x, int y)
{
    int posi = channel * kernel_size * kernel_size + y * kernel_size + x; 
    if(posi < in_channels * kernel_size * kernel_size){
        return posi;
    }
    else {
        throw "out of index";
    }
}

int Conv2D::out_dimension()
{
    return out_channels * out_H * out_W;
}

int Conv2D :: outH()
{
    return this -> out_H;
}

int Conv2D :: outW()
{
    return this -> out_W;
}

int Conv2D :: outchannel()
{
    return this -> out_channels;
}

// // 原始代码
// vector<string> split(const string &str, const string &pattern)
// {
//     vector<string> res;
//     if(str == " ")
//         return res;
//     //在字符串末尾也加入分隔符，方便截取最后一段
//     string strs = str; /*+ pattern;*/
//     size_t pos = strs.find(pattern);

//     while(pos != strs.npos)
//     {
//         string temp = strs.substr(0, pos);
//         res.push_back(temp);
//         //去掉已分割的字符串,在剩下的字符串中进行分割
//         strs = strs.substr(pos+pattern.size(), strs.size());
//         pos = strs.find(pattern);
//     }

//     return res;
// }

//  修改后的代码
vector<string> split(const string &str, const string &pattern)
{
    vector<string> res;
    if (str.empty())
        return res;

    string strs = str; // 原始字符串
    size_t pos = strs.find(pattern);

    while (pos != string::npos)
    {
        string temp = strs.substr(0, pos);
        res.push_back(temp);
        strs = strs.substr(pos + pattern.size()); // 更新剩余字符串
        pos = strs.find(pattern);
    }

    // 添加最后一段（如果有的话）
    if (!strs.empty())
    {
        res.push_back(strs);
    }

    return res;
}
