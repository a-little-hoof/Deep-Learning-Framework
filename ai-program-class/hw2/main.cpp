#include "layer.h"
#include "tensor.h"

int main(){
    //test init 
    printf("testing init...\n\n");
    std::vector<int> shape0 = {2, 3};
    Tensor a(shape0, "GPU");
    a.fill_(0);
    a.print();
    printf("a\n\n");
    matrix_init(a);
    a.print();
    printf("a\n\n");

    //test fc_forward
    printf("testing fc_forward...\n\n");
    std::vector<int> shape1 = {2, 3};
    std::vector<int> shape2 = {3, 4};
    std::vector<int> shape3 = {4};
    Tensor X(shape1, "CPU");
    Tensor W(shape2, "CPU");
    Tensor b(shape3, "GPU");
    Tensor Y(std::vector<int>{2, 4}, "GPU");
    for (int i = 0; i < 6; i++){
        X.data[i] = i+1;
    }
    for (int i = 0; i < 12; i++){
        W.data[i] = i+7;
    }
    X.gpu();
    W.gpu();
    b.fill_(1);
    X.print();
    printf("X\n\n");
    W.print();
    printf("W\n\n");
    b.print();
    printf("b\n\n");
    fc_forward(X, W, b, Y);
    Y.print();
    printf("Y\n");
    printf("\n\n");

    //test fc_backward
    printf("testing fc_backward...\n\n");
    Tensor dY(std::vector<int>{2, 4}, "GPU");
    Tensor dW(shape2, "GPU");
    Tensor db(shape3, "GPU");
    Tensor dX(shape1, "GPU");
    dY.fill_(1.0);
    dW.fill_(0.0);
    dX.fill_(0.0);
    fc_backward(dY, X, W, dW, dX);
    dW.print();
    printf("dW\n\n");
    dX.print();
    printf("dX\n\n");
    printf("\n\n");

    //test conv_forward
    printf("testing conv_forward...\n\n");
    std::vector<int> shape4 = {2, 3, 2, 2};
    std::vector<int> shape5 = {1, 3, 3, 3};
    Tensor X1(shape4, "GPU");
    Tensor W1(shape5, "GPU");
    Tensor Y1(std::vector<int>{2, 1, 2, 2}, "GPU");
    X1.fill_(1.0);
    W1.fill_(1.0);
    Y1.fill_(1.0);
    Y1.print();
    printf("Y1\n\n");
    W1.print();
    printf("W1\n\n");
    conv_forward(X1, W1, Y1);
    Y1.print();
    printf("Y1\n");
    printf("\n\n");

    //test conv_backward
    printf("testing conv_backward...\n\n");
    Tensor dY1(std::vector<int>{2, 1, 2, 2}, "GPU");
    Tensor dW1(shape5, "GPU");
    Tensor dX1(shape4, "GPU");
    dY1.fill_(1.0);
    dW1.fill_(0.0);
    dX1.fill_(0.0);
    conv_backward(dY1, X1, W1, dW1, dX1);
    dW1.print();
    printf("dW1\n\n");
    dX1.print();
    printf("dX1\n");
    printf("\n\n");

    //test maxpool_forward
    printf("testing maxpool_forward...\n\n");
    std::vector<int> shape6 = {1, 3, 4, 4};
    Tensor X2(shape6, "CPU");
    Tensor Y2(std::vector<int>{1, 3, 2, 2}, "GPU");
    Tensor mask(std::vector<int>{1, 3, 2, 2}, "GPU");
    for (int i = 0; i < 48; i++){
        X2.data[i] = i+1;
    }
    X2.gpu();
    X2.print();
    printf("X2\n\n");
    Y2.fill_(1.0);
    mask.fill_(-1.0);
    maxpool_forward(X2, Y2, mask);
    mask.print();
    printf("mask\n\n");
    Y2.print();
    printf("Y2\n");
    printf("\n\n");

    //test maxpool_backward
    printf("testing maxpool_backward...\n\n");
    Tensor dY2(std::vector<int>{1, 3, 2, 2}, "GPU");
    Tensor dX2(shape6, "GPU");
    dY2.fill_(1.0);
    dX2.fill_(0.0);
    maxpool_backward(dY2, mask, dX2);
    dX2.print();
    printf("dX2\n");
    printf("\n\n");

    //test softmax_forward
    printf("testing softmax_forward...\n\n");
    std::vector<int> shape7 = {2, 3};
    Tensor X3(shape7, "CPU");
    Tensor Y3(std::vector<int>{2, 3}, "GPU");
    for (int i = 0; i < 6; i++){
        X3.data[i] = i+1;
    }
    X3.gpu();
    X3.print();
    printf("X3\n\n");
    Y3.fill_(1.0);
    softmax_forward(X3, Y3);
    Y3.print();
    printf("Y3\n");
    printf("\n\n");

    //test cross_entropy_forward
    printf("testing cross_entropy_forward...\n\n");
    std::vector<int> shape8 = {2, 3};
    Tensor X4(shape8, "CPU");
    Tensor label(std::vector<int>{2, 3}, "CPU");
    Tensor loss(std::vector<int>{1}, "GPU");
    for (int i = 0; i < 6; i++){
        X4.data[i] = i+1;
    }
    label.data[0] = 0;
    label.data[1] = 1;
    label.data[2] = 0;
    label.data[3] = 0;
    label.data[4] = 0;
    label.data[5] = 1;
    X4.gpu();
    label.gpu();
    X4.print();
    printf("X4\n\n");
    label.print();
    printf("label\n\n");
    loss.fill_(1.0);
    cross_entropy_forward(X4, label, loss);
    loss.print();
    printf("loss\n");
    printf("\n\n");

    //test cross_entropy_with_softmax_backward
    printf("testing cross_entropy_with_softmax_backward...\n\n");
    Tensor L(std::vector<int>{1}, "GPU");
    L.fill_(1.0);
    Tensor dX4(shape8, "GPU");
    dX4.fill_(0.0);
    cross_entropy_with_softmax_backward(L, X4, label, dX4);
    dX4.print();
    printf("dX4\n");


    
    return 0;
}