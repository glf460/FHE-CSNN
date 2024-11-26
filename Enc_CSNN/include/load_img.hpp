#include <string>
#include <vector>
#include <fstream>
#include "Layer.hpp"
#include "tfhe_import.hpp"

using namespace std;

vector<vector<int>> load_img_255(string);

vector<int> load_label(string);

int discretize(int value);