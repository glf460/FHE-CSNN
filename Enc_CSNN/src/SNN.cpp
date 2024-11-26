#include "../include/SNN.hpp"
#include "lwe-functions.h"
#include "tfhe_core.h"
#include <vector>
#include <chrono>  // 引入 chrono 库来进行时间计算
#include <fstream> // 引入 ofstream 来写入文件
#include <sstream> // 引入 stringstream 用于文件名生成

using namespace std::chrono; // 方便使用 std::chrono::steady_clock

//                                      scale_down
// Enc_Snn::Enc_Snn(int scale) : fc1(3 * 32 * 32, 10), ac1(10, 1 * scale)
// Enc_Snn::Enc_Snn(int scale) : fc1(28 * 28, 10), ac1(10, 1 * scale)
Enc_Snn::Enc_Snn(int scale) : fc1(28*28, 10), ac1(10, 1 * scale)
{
    this->scale_down = scale; //  将传入的参数 scale 赋值给 Enc_Snn 对象的成员变量 scale_down。this 指针指向当前对象实例
}

// 定义 Enc_Snn 类的成员函数 enc_forward。它接受两个参数: LweSample * x  和 int x_dim
// LweSample * x：指向 LweSample 类型的指针，表示输入的加密数据
// int x_dim：输入数据的维度
// 返回类型是 LweSample *，指向加密输出数据的指针
// 函数的作用是对输入的加密数据进行前向传播，首先通过一个全连接层，然后通过一个激活层，最后返回处理后的加密输出数据
LweSample * Enc_Snn:: enc_forward(LweSample * x, int x_dim){
    LweSample * out = this->fc1.enc_forward(x, x_dim);
    out = this->ac1.enc_forward(out, 10);

    return out;
}

// 定义 Enc_Snn 类的成员函数 save，接受一个 string 类型的参数 posi_weight，没有返回值
// 函数的作用是调用 fc1 对象的 save 方法，将全连接层的权重保存到指定的位置 posi_weight
void Enc_Snn:: save(string posi_weight){
    // 调用 fc1 对象的 save 方法，将 posi_weight 作为参数传递给它，以便保存当前层的权重到指定的位置
    this->fc1.save(posi_weight);
}

// 定义 Enc_Snn 类的成员函数 load，接受一个 string 类型的参数 posi_weight，没有返回值
// 函数的作用是调用 fc1 对象的 load 方法，使用指定的位置 posi_weight 和 scale_down 的一半来加载全连接层的权重
void Enc_Snn:: load(string posi_weight){
    this->fc1.load(posi_weight, this->scale_down / 2);
}

// 定义 Enc_Snn 类的成员函数 show_scale_down，不接受任何参数，返回类型为 int
// 函数的作用是返回 Enc_Snn 对象的 scale_down 成员变量的值
int Enc_Snn:: show_scale_down(){
    return this->scale_down;
}

// 定义 Enc_Snn 类的成员函数 reset_voltyy，不接受任何参数，返回类型为 void
// 函数的作用是调用 ac1 对象的 enc_reset_state 方法，以重置激活层的状态
void Enc_Snn:: resrt_volty(){
    this->ac1.enc_reset_state();
}

// 定义 Enc_SNNDF 类的构造函数，接受一个 int 类型的参数 scale
// 函数的作用是初始化 Enc_SNNDF 类的对象，并设置其 scale_down 成员变量的值, 同时通过初始化列表来初始化 fc1、ac1、fc2 和 ac2 四个成员变量
Enc_SNNDF::Enc_SNNDF(int scale) :  // 初始化列表
// fc1(96 * 96, 30), 
fc1(28 * 28, 30),       
ac1(30, 1.0 * scale),  
// fc2(30,100),
fc2(30, 10),            
ac2(2, 1.0 * scale)    
{
    this->scale_down = scale;
}

// 函数的作用是对输入数据 x 进行前向传播，通过 fc1、ac1、fc2 和 ac2 层，进行解密和重新加密处理，最终返回处理后的结果
LweSample * Enc_SNNDF:: enc_forward(LweSample * x, int x_dim){
    LweSample * out = this->fc1.enc_forward(x, x_dim);
    out = this->ac1.enc_forward(out, 30);
    for (int i = 0; i < 30; i++) {
        auto score = lweSymDecrypt(out + i, secret->lwe_key, msg_space);
        // cout << score << " " ;
        lweNoiselessTrivial(out+i, score, in_out_params);     //测试用
    }
    // cout << endl;

    out = this->fc2.enc_forward(out, 30);
    out = this->ac2.enc_forward(out, 10);

    return out;
}

// save是 Enc_SNNDF类的成员函数。参数layer，int类型，表示层的数量。参数positions，vector<string>类型，表示保存文件的路径列表
// 函数的作用是将模型的权重保存到指定的文件位置。如果提供的文件位置数量与层的数量不匹配，则抛出异常
void Enc_SNNDF:: save(int layers, vector<string> positions){
    // 检查 positions 向量的大小是否等于 this->layers。this->layers 表示 Enc_SNNDF 对象中的层
    if(positions.size() != this->layers){
        throw "nums of files doesnot match layers";
    }
    this->fc1.save(positions[0]);
    this->fc2.save(positions[1]);
}

// load Enc_SNNDF类的成员函数。参数layer，int类型，表示层的数量。参数positions，vector<string>类型，表示保存文件的路径列表
// 函数的作用是从指定的文件位置加载模型的权重。如果提供的文件位置数量与层的数量不匹配，则抛出异常
void Enc_SNNDF:: load(int layers, vector<string> positions){
    if(positions.size() != this->layers){
        throw "nums of files doesnot match layers";
    }
    // // 调整加载权重的缩放比例
    this->fc1.load(positions[0], this->scale_down / 2);
    this->fc2.load(positions[1], this->scale_down / 2);
}

// 定义 Enc_SNNDF 类的成员函数 show_scale_down，返回一个 int 类型的值
// 函数的作用是返回当前 Enc_SNNDF 对象的 scale_down 值，用于查看或获取该对象的 scale_down 值
int Enc_SNNDF:: show_scale_down(){
    return this->scale_down;
}

// 定义 Enc_SNNDF 类的成员函数 resrt_volty，没有返回值
// 函数的作用是重置当前 Enc_SNNDF 对象中两个激活层（ac1 和 ac2）的状态。这样可以确保在下一次使用这些激活层时，它们处于初始状态
void Enc_SNNDF:: resrt_volty(){
    this->ac1.enc_reset_state();
    this->ac2.enc_reset_state();
}

// Enc_CSNN 是类名，Enc_CSNN::Enc_CSNN 是构造函数
// 构造函数参数包括：scale, kernel_nums, Vth, kernelsize, stride, pad, poolingsize, hidden_dim, LIF, 和 tau
// 构造函数的作用是创建并初始化 Enc_CSNN 类的一个对象，同时将传入的参数赋值给相应的成员变量，并初始化类中的各个层
Enc_CSNN::Enc_CSNN(int scale ,int kernel_nums, int Vth, int kernelsize, int stride, int pad, int poolingsize, int hidden_dim, bool LIF, int tau):
// conv(kernelsize, kernel_nums, stride, pad, 1, 28, 28),     // 1 通道
conv(kernelsize, kernel_nums, stride, pad, 1, 28, 28),  // 3 通道
lif1(conv.out_dimension(), 1*scale, LIF, tau), //10*28*28
mp(poolingsize, poolingsize, false, conv.outchannel(), conv.outH(), conv.outW()),//10*14*14
fc(mp.out_dimension(), hidden_dim),
lif2(hidden_dim, 1*scale, LIF, tau),
fc2(hidden_dim, 10),
lif3(10, 1*scale, LIF, tau)
// fc2(hidden_dim, 100),
// lif3(100, 1*scale, LIF, tau)
{ // 函数体，设置类成员变量 scale、hidden_dim 和 decay_tau 的值
    
    this->scale = scale;
    this->hidden_dim = hidden_dim;
    this->decay_tau = tau;
}

bool Enc_CSNN::load(vector<string> path, bool decayinput)
{
    // 检查 decayinput 是否为真。如果为真，则执行以下代码块，否则执行 else 块中的代码
    if (decayinput)
    {
        conv.load(path[0], this -> decay_tau * this -> scale * 2);
        fc.load(path[1], this -> decay_tau * this -> scale / 4);
        fc2.load(path[2], this -> decay_tau * this -> scale);
    }
    else
    {
        conv.load(path[0], this -> scale * 2);
        fc.load(path[1], this -> scale / 4);
        fc2.load(path[2], this -> scale);
    }
    return true;
}

// LweSample* Enc_CSNN::enc_forward(LweSample* input, int channel, int length, int width)
// {
//     auto output = conv.forward(input, channel, length, width);
//     output = lif1.enc_forward(output, conv.out_dimension());

//     // cout << conv.out_dimension() << endl;

//     output = mp.enc_forward(output, conv.outchannel(), conv.outH(), conv.outW());

//     output = fc.enc_forward(output, mp.out_dimension());

//     output = lif2.enc_forward(output, hidden_dim);

//     output = fc2.enc_forward(output, hidden_dim);

//     output = lif3.enc_forward(output, 10);       // 类别数
//     // output = lif3.enc_forward(output, 100);

//     return output;

// }
LweSample *Enc_CSNN::enc_forward(LweSample *input, int channel, int length, int width)
{
    // 获取当前时间并生成文件名
    std::stringstream filename;
    filename << "/home/glf/C++Projects/run_time/run_time_2.txt";

    // 创建文件输出流对象
    std::ofstream out_file;
    out_file.open(filename.str(), std::ios::out | std::ios::app); // 以追加模式打开文件

    if (!out_file.is_open())
    {
        std::cerr << "Error opening file!" << std::endl;
        return nullptr;
    }

    // 记录卷积层的运行时间
    auto start = steady_clock::now();
    auto output = conv.forward(input, channel, length, width); // 执行卷积操作
    auto end = steady_clock::now();
    auto conv_duration = duration_cast<seconds>(end - start); // 计算时间差并转换为秒
    out_file << "Conv layer time: " << conv_duration.count() << " seconds" << std::endl;

    // 记录第一个 LIF 层的运行时间
    start = steady_clock::now();
    output = lif1.enc_forward(output, conv.out_dimension()); // 执行第一个 LIF 层的前向传播
    end = steady_clock::now();
    auto lif1_duration = duration_cast<seconds>(end - start);
    out_file << "LIF1 layer time: " << lif1_duration.count() << " seconds" << std::endl;

    // 记录池化层的运行时间
    start = steady_clock::now();
    output = mp.enc_forward(output, conv.outchannel(), conv.outH(), conv.outW()); // 执行池化操作
    end = steady_clock::now();
    auto mp_duration = duration_cast<seconds>(end - start);
    out_file << "AvgPool layer time: " << mp_duration.count() << " seconds" << std::endl;

    // 记录第一个全连接层的运行时间
    start = steady_clock::now();
    output = fc.enc_forward(output, mp.out_dimension()); // 执行第一个全连接层的前向传播
    end = steady_clock::now();
    auto fc_duration = duration_cast<seconds>(end - start);
    out_file << "FC layer time: " << fc_duration.count() << " seconds" << std::endl;

    // 记录第二个 LIF 层的运行时间
    start = steady_clock::now();
    output = lif2.enc_forward(output, hidden_dim); // 执行第二个 LIF 层的前向传播
    end = steady_clock::now();
    auto lif2_duration = duration_cast<seconds>(end - start);
    out_file << "LIF2 layer time: " << lif2_duration.count() << " seconds" << std::endl;

    // 记录第二个全连接层的运行时间
    start = steady_clock::now();
    output = fc2.enc_forward(output, hidden_dim); // 执行第二个全连接层的前向传播
    end = steady_clock::now();
    auto fc2_duration = duration_cast<seconds>(end - start);
    out_file << "FC2 layer time: " << fc2_duration.count() << " seconds" << std::endl;

    // 记录第三个 LIF 层的运行时间
    start = steady_clock::now();
    output = lif3.enc_forward(output, 10); // 执行第三个 LIF 层的前向传播
    end = steady_clock::now();
    auto lif3_duration = duration_cast<seconds>(end - start);
    out_file << "LIF3 layer time: " << lif3_duration.count() << " seconds" << std::endl;

    // 关闭文件
    out_file.close();

    return output;
}

// 定义 Enc_CSNN 类的成员函数 reset_volty，没有返回值且不接受参数
void Enc_CSNN::reset_volty()
{
    lif1.enc_reset_state();
    lif2.enc_reset_state();
}

// 定义 Enc_CSNN 类的成员函数 show_max_value，没有返回值且不接受参数
// 调用 lif1、lif2 和 lif3 的 show_max 方法，将它们的最大值添加到输出流
// lif1、lif2、lif3 都是 Enc_CSNN 类的成员
void Enc_CSNN :: show_max_value(){
    // 
    cout << "max of SNN layer 1,2,3 are: " << lif1.show_max() << " " << lif2.show_max() << " " << lif3.show_max() << endl; // 
}
