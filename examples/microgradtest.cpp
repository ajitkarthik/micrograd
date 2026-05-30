#include <iostream>

#include "value.hpp"

int main() {
    {
        Value x1 = Value(2.0, "x1");
        Value x2 = Value(0.0, "x2");
        Value w1 = Value(-3.0, "w1");
        Value w2 = Value(1.0, "w2");
        Value b = Value(6.8813735870, "b");

        // x1w1 + x2w2
        Value x1w1 = x1 * w1;
        x1w1.label("x1*w1");
        Value x2w2 = x2 * w2;
        x2w2.label("x2*w2");
        Value x1w1x2w2 = x1w1 + x2w2;
        x1w1x2w2.label("x1*w1 + x2*w2");
        Value n = x1w1x2w2 + b;
        n.label("n");

        Value o = n.tanh();
        o.label("o");
        o.grad(1.0);
        o.backward();
        std::cout << o;
    }

    // {
    //     Value a = Value(3.0, "a");
    //     Value b = a + a;
    //     b.label("b");
    //     b.grad(1.0);
    //     b.backward();
    //     std::cout << b;
    // }

    // {
    //     //   c = a + b
    //     //   d = c * k     // k is some Value(2.0)
    //     //   e = c + d     // c is now shared between two non-leaf consumers
    //     Value a = Value(1.0, "a");
    //     Value b = Value(2.0, "b");
    //     Value c = a + b;
    //     c.label("c");
    //     Value k = Value(2.0, "k");
    //     Value d = c * k;
    //     d.label("d");
    //     Value e = c + d;
    //     e.label("e");
    //     e.grad(1.0);
    //     e.backward();
    //     std::cout << e;
    // }
}