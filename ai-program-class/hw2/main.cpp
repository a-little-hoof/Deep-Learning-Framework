#include "layer.h"
#include "tensor.h"

int main(){
    //test fc_forward
    std::vector<int> shape1 = {2, 3};
    std::vector<int> shape2 = {3, 4};
    std::vector<int> shape3 = {4};
    Tensor X(shape1, "GPU");
    Tensor W(shape2, "GPU");
    Tensor b(shape3, "GPU");
    Tensor Y(std::vector<int>{2, 4}, "GPU");
    X.fill_(1.0);
    W.fill_(1.0);
    b.fill_(1.0);
    X.print();
    printf("X\n");
    W.print();
    printf("W\n");
    b.print();
    printf("b\n");
    fc_forward(X, W, b, Y);
    Y.print();
    printf("Y\n");

    //test fc_backward
    Tensor dY(std::vector<int>{2, 4}, "GPU");
    Tensor dW(shape2, "GPU");
    Tensor db(shape3, "GPU");
    Tensor dX(shape1, "GPU");
    dY.fill_(1.0);
    dW.fill_(0.0);
    dX.fill_(0.0);
    fc_backward(dY, X, W, dW, dX);
    dW.print();
    printf("dW\n");
    dX.print();
    printf("dX\n");
    
    return 0;
}