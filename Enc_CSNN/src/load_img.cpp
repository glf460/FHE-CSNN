#include "../include/load_img.hpp"
#include <string>

//用于从指定文件中读取图像数据，并将其转换为二维整数向量
vector<vector<int>> load_img_255(string posi_img)
{
    ifstream file;
    file.open(posi_img, ios_base::in);
    string line;
    vector<int> img;
    vector<vector<int>> img_all;
    while (getline(file, line)) {
        auto split_line = split(line, " ");
        for(int j = 0; j < split_line.size(); j++){
            img.push_back(stoi(split_line[j]));
        }
        img_all.push_back(img);
        img.clear();
    }

    return img_all;
}

vector<int> load_label(string posi_label) {
    ifstream file;
    file.open(posi_label, ios_base::in);
    string line;
    vector<int> label;
    while (getline(file, line)) {
        label.push_back(stoi(line));
    }

    return label;
}



// // 用于从指定文件中读取 3 通道 32 x 32 图像数据，每一行是一张图片，共3072列，转换为三维整数向量
// vector<vector<vector<int>>> load_img_255(string posi_img)
// {
//     ifstream file;
//     file.open(posi_img, ios_base::in);
//     string line;
//     vector<vector<vector<int>>> img_all(3, vector<vector<int>>(32, vector<int>(32))); // 三个通道分别保存 32x32 图像

//     while (getline(file, line))
//     {
//         auto split_line = split(line, " "); // 假设每列的值以空格分隔
//         if (split_line.size() != 3072)
//         {
//             throw "Invalid image data format, expected 3072 columns.";
//         }

//         // 解析 3072 列数据到 3 个通道，每个通道对应 32x32
//         for (int i = 0; i < 1024; i++)
//         {
//             int row = i / 32;                                  // 行号
//             int col = i % 32;                                  // 列号
//             img_all[0][row][col] = stoi(split_line[i]);        // R 通道
//             img_all[1][row][col] = stoi(split_line[i + 1024]); // G 通道
//             img_all[2][row][col] = stoi(split_line[i + 2048]); // B 通道
//         }
//     }

//     return img_all;
// }

// // 读取标签文件并将其转换为一维整数向量
// vector<int> load_label(string posi_label)
// {
//     ifstream file;
//     file.open(posi_label, ios_base::in);
//     string line;
//     vector<int> label;
//     while (getline(file, line))
//     {
//         label.push_back(stoi(line));
//     }

//     return label;
// }
