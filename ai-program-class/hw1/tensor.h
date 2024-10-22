#ifndef TENSOR_H
#define TENSOR_H


#include <vector>
#include <string>
using namespace std;

class Tensor{
    public:
        vector<int> shape;
        string device;
        float* data;

        Tensor(vector<int> s, string d);

        ~Tensor();

        void cpu();

        void gpu();

        int get_size();
};

#endif

