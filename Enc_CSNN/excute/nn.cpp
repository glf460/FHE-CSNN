// Includes
#include <ctime>
#include <ios>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>
#include <sys/time.h>
#include <sys/types.h>
#include <dirent.h>

// Multi-processing
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

// tfhe-lib
#include "../include/encode.hpp"
#include "../include/tfhe_import.hpp"
#include "../include/SNN.hpp"

using namespace std;

// 数据及标签
#define IMG_PATH "../data/MNIST/MNIST.txt"
#define LABEL_PATH "../data/MNIST/MNIST_label.txt"
// #define IMG_PATH "/home/glf/C++Projects/dataset/MNIST/shuffled_image_pixels.txt"
// #define LABEL_PATH "/home/glf/C++Projects/dataset/MNIST/shuffled_image_labels.txt"

// 模型参数
#define WEIGHTS_PATH1 "/home/glf/C++Projects/model_parameters/MNIST/conv1_weight_tau(2.0).txt"
#define WEIGHTS_PATH2 "/home/glf/C++Projects/model_parameters/MNIST/fc1_weight_tau(2.0).txt"
#define WEIGHTS_PATH3 "/home/glf/C++Projects/model_parameters/MNIST/fc2_weight_tau(2.0).txt"

// #define LOG_PATH "../logs/Tumor/"
#define LOG_PATH "/home/glf/C++Projects/logs/MNIST/tau(2.0)/"

// τ 参数
#define TAU 2.0

// decay_input = ture or false
#define DECAYINPUT false

// Generate gate bootstrapping parameters for FHE_NN
vector<vector<int>> load_img_255(string);
vector<int> load_label(string posi_label);

int main(int argc, char **argv)
{
    // 输出文件名
    string fullPath = WEIGHTS_PATH3;
    size_t pos = fullPath.find_last_of("/\\"); // 找到最后一个斜杠的位置
    string fileName = (pos == string::npos) ? fullPath : fullPath.substr(pos + 1);
    cout << "WEIGHTS_PATH3: " << fileName << endl;

    auto img_all = load_img_255(IMG_PATH);
    auto label_all = load_label(LABEL_PATH);
    int img_nums = img_all.size();
    int img_dim = img_all[0].size();

    int out_dim = 10;    // 类别数

    int count_right = 0;
    int T = 4;

    // θ 参数
    int scale_down = 30;

    // for(int i = 1;i <= argc; i++){
    //     if (i == 1) {
    //         T =  stoi(argv[i]);
    //     }
    //     else if (i == 2) {
    //         scale_down = stoi(argv[i]);
    //     }
    // }

    time_t now = time(nullptr);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now), "%F-%T");
    string path = LOG_PATH + ss.str() + "-T" + to_string(T) + "-tau2.0-theta_record_run_time" + to_string(scale_down);
    system(("mkdir -p " + path).c_str());

    vector<string> WEIGHTS_PATH;
    WEIGHTS_PATH.push_back(WEIGHTS_PATH1);
    WEIGHTS_PATH.push_back(WEIGHTS_PATH2);
    WEIGHTS_PATH.push_back(WEIGHTS_PATH3);

    // 当false的时候，最后一个那个参数就无效了，随便填啥都行，tau = ∞，为false
    Enc_CSNN net(scale_down, 10, 1, 3, 1, 1, 2, 160, true, TAU);

    net.load(WEIGHTS_PATH, DECAYINPUT);

    ofstream logreslut(path + "/result.txt", ios_base::out);
    //! malloc space for output and img_encode
    // LweSample * output = new_LweSample_array(10, in_out_params);
    LweSample *output = new_LweSample_array(out_dim, in_out_params);
    LweSample *enc_img_encode = new_LweSample_array(img_dim, in_out_params);

    clock_t begin, end;
    begin = clock();
    int max_nums = 200;

    for (int j = 0; j < max_nums; j++)
    {
        // for (int j = 0; j < img_nums; j++)
        //! encrypt the img
        LweSample *enc_img = new_LweSample_array(img_dim, in_out_params);
        for (int i = 0; i < img_dim; i++)
        {
            Torus32 mu = modSwitchToTorus32(floor(img_all[j][i] / 255.00 * 4), msg_space);
            // cout<<round(img_all[j][i]/255.00)<<endl;
            // lweSymEncrypt(enc_img+i, mu, alpha, secret->lwe_key);
            lweNoiselessTrivial(enc_img + i, mu, in_out_params);
        }

        // for (int i = 0; i < 10; i++) {
        for (int i = 0; i < out_dim; i++)
        {
            lweNoiselessTrivial(output + i, 0, in_out_params);
        }

        for (int t = 0; t < T; t++)
        {

            // auto temp = net.enc_forward(enc_img, 3, 32, 32);
            auto temp = net.enc_forward(enc_img, 1, 28, 28);

            for (int i = 0; i < out_dim; i++)
            {
                lweAddTo(output + i, temp + i, in_out_params);
            }
        }

        net.reset_volty();

        int classify = 0;
        int max_score = -1;

        cout << "output = ";

        for (int i = 0; i < out_dim; i++)
        {
            auto score = lweSymDecrypt(output + i, secret->lwe_key, msg_space);
            score = modSwitchFromTorus32(score, msg_space);
            score = score < msg_space / 2 ? score : score - msg_space;

            cout << score << "   ";

            if (score > max_score)
            {
                max_score = score;
                classify = i;
            }
        }
        logreslut << classify << endl;

        cout << "   label is " << label_all[j] << endl;

        if (classify == label_all[j])
        {
            count_right++;
        }

        // delete_LweSample_array(10, output);
    }

    logreslut.close();
    end = clock();

    net.show_max_value();

    cout << "the accurary is " << double(count_right) / max_nums << endl;

    const double clocks2seconds = 1. / CLOCKS_PER_SEC;

    ofstream logfile(path + "/params.txt", ios_base::out);
    logfile << "msg_space = " << msg_space << endl;
    logfile << "N = " << SEC_PARAMS_N << endl;
    logfile << "scale sown = " << scale_down << endl;
    logfile << "T = " << T << endl;
    logfile << "TAU = " << TAU << endl;
    logfile << "time/per img = " << double(end - begin) / max_nums * clocks2seconds << endl;
    logfile << "accurary = " << double(count_right) / max_nums << endl;
    logfile.close();

    return 0;
}
