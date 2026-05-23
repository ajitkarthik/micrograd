#include <cstddef>
#include <iostream>
#include <memory>
#include <numbers>
#include <ostream>
#include <string>
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
                   std::unordered_set<const Node*>* visited = nullptr) const {
            std::unordered_set<const Node*> local_visited;
            if (!visited) visited = &local_visited;
            os << std::string(depth * 2, ' ');
            os << "Value(";
            if (!label.empty()) os << label << " : ";
            if (!visited->insert(this).second) {
                os << "...already printed)\n";
                return;
            }
            os << "data=" << data;
            if (!opstring.empty()) os << ", opstring=" << opstring;
            os << ", op=" << static_cast<int>(op);
            os << ", grad=" << grad;
            os << ")\n";
            for (const auto& p : prev) {
                p->print(os, depth + 1, visited);
            }
        }

        // gradient computation - See Karpathy's micrograd video for an explanation of this.
        void backward(void) {
            std::cout << "Computing backward on " << label << std::endl;
            switch (op) {
                case Node::Op::TANH:
                    prev[0]->grad += (1.0 - pow(data, 2)) * grad;
                    break;
                case Node::Op::ADD:
                    prev[0]->grad += 1.0 * grad;
                    prev[1]->grad += 1.0 * grad;
                    break;
                case Node::Op::MUL:
                    prev[0]->grad += prev[1]->data * grad;
                    prev[1]->grad += prev[0]->data * grad;
                    break;
                case Node::Op::LEAF:
                    break;
            }
            for (const auto& p : prev) {
                p->backward();
            }
        }
    };

    std::shared_ptr<Node> node_;

    // non-leaf (operation) constructor
    Value(double data, std::vector<std::shared_ptr<Node>> parents, Node::Op op,
          const std::string& opstring, const std::string& label = "") {
        node_ = std::make_shared<Node>(data, parents, op, opstring, label);
    }

   public:
    // ===getters===
    std::shared_ptr<Node> node() const { return node_; }

    // ===setters===
    void label(const std::string& label) { node_->label = label; }
    void grad(double grad) { node_->grad = grad; }

    // ===Constructors===
    // leaf node constuctor
    Value(double data, const std::string& label) {
        // for leaf value node, initialize with empty parents
        node_ = std::make_shared<Node>(data, std::vector<std::shared_ptr<Node>>{}, Node::Op::LEAF,
                                       "", label);
    }

    // constructor to go from Node object to Value object
    Value(std::shared_ptr<Node> n) { node_ = std::move(n); }

    // ===Operator overloads===
    // + operator
    Value operator+(const Value& other) const {
        return Value(node_->data + other.node_->data, {node_, other.node_}, Node::Op::ADD, "+");
    }

    // tanh operator
    Value tanh() const {
        using std::pow;
        using std::numbers::e;
        return Value((pow(e, 2 * node_->data) - 1) / (pow(e, 2 * node_->data) + 1), {node_},
                     Node::Op::TANH, "tanh");
    }

    // * operator
    Value operator*(const Value& other) const {
        return Value(node_->data * other.node_->data, {node_, other.node_}, Node::Op::MUL, "*");
    }

    void backward() { node_->backward(); }
};

std::ostream& operator<<(std::ostream& os, const Value& v) {
    v.node()->print(os);
    return os;
}

int main(void) {
    {
        Value x1 = Value(2.0, "x1");
        Value x2 = Value(0.0, "x2");
        Value w1 = Value(-3, "w1");
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
}