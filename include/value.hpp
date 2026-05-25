#pragma once

#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>

class Value {
   private:
    struct Node {
        double data;
        double grad;
        std::vector<std::shared_ptr<Node>> prev;
        enum class Op { LEAF, ADD, MUL, TANH };
        Op op;
        std::string opstring;
        std::string label;
        Node(double data, std::vector<std::shared_ptr<Node>> prev = {}, const Op op = Op::LEAF,
             const std::string& opstring = "", const std::string& label = "")
            : data(data), grad(0), prev(prev), op(op), opstring(opstring), label(label) {}

        void print(std::ostream& os, int depth = 0,
                   std::unordered_set<const Node*>* visited = nullptr) const;
        void printNode(std::ostream& os) const;
        // gradient computation - See Karpathy's micrograd video for an explanation of this.
        void backward(void);
        void backward_local(void);
        void DFSVisit(Node* curr, std::vector<Node*>& visited, std::stack<Node*>& stack) const;
    };

    std::shared_ptr<Node> node_;

    // non-leaf (operation) constructor
    Value(double data, std::vector<std::shared_ptr<Node>> parents, Node::Op op,
          const std::string& opstring, const std::string& label = "");

   public:
    // ===getters===
    std::shared_ptr<Node> node() const;

    // ===setters===
    void label(const std::string& label);
    void grad(double grad);

    // ===Constructors===
    // leaf node constuctor
    Value(double data, const std::string& label);

    // constructor to go from Node object to Value object
    Value(std::shared_ptr<Node> n);

    // ===Operator overloads===
    // + operator
    Value operator+(const Value& other) const;
    // tanh operator
    Value tanh() const;

    // * operator
    Value operator*(const Value& other) const;

    void backward();
};

std::ostream& operator<<(std::ostream& os, const Value& v);