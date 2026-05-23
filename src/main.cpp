#include <cstddef>
#include <iostream>
#include <memory>
#include <numbers>
#include <ostream>
#include <string>
#include <vector>

class Value {
   private:
    struct Node {
        float data;
        float grad;
        std::vector<std::shared_ptr<Node>> prev;
        std::string op;
        std::string label;
        Node(float data, std::vector<std::shared_ptr<Node>> prev = {}, const std::string& op = "",
             const std::string& label = "")
            : data(data), grad(0), prev(prev), op(op), label(label) {}

        void print(std::ostream& os, int depth = 0) const {
            os << std::string(depth * 2, ' ');
            os << "Value(";
            if (!label.empty()) os << label << " : ";
            os << "data=" << data;
            if (!op.empty()) os << ", op=" << op;
            os << ")\n";
            for (const auto& child : prev) {
                child->print(os, depth + 1);
            }
        }
    };

    std::shared_ptr<Node> node_;

   public:
    // getters
    float data() const { return node_->data; }
    std::shared_ptr<Node> node() const { return node_; }
    std::vector<std::shared_ptr<Node>> parents() const { return node_->prev; }

    // setters
    void label(const std::string& label) { node_->label = label; }

    // Constructors
    // leaf node constuctor
    Value(float data, const std::string& label) {
        // for leaf value node, initialize with empty parents
        node_ = std::make_shared<Node>(data, std::vector<std::shared_ptr<Node>>{}, "", label);
    }

    // non-leaf (operation) constructor
    Value(float data, std::vector<std::shared_ptr<Node>> parents, const std::string& op,
          const std::string& label = "") {
        node_ = std::make_shared<Node>(data, parents, op, label);
    }

    // constructor to go from Node object to Value object
    Value(std::shared_ptr<Node> n) { node_ = std::move(n); }

    // + operator
    Value operator+(const Value& other) const {
        return Value(data() + other.data(), {node_, other.node_}, "+");
    }

    // tanh operator
    Value tanh() const {
        using std::pow;
        using std::numbers::e;
        return Value((pow(e, 2 * data()) - 1) / (pow(e, 2 * data() + 1)), {node_}, "tanh");
    }

    // * operator
    Value operator*(const Value& other) const {
        return Value(data() * other.data(), {node_, other.node_}, "*");
    }
};

std::ostream& operator<<(std::ostream& os, const Value& v) {
    v.node()->print(os);
    return os;
}

int main(void) {
    float h = 0.001;

    Value a = Value(2.0, "a");
    Value b = Value(-3.0, "b");
    Value c = Value(10.0, "c");
    Value e = a * b;
    e.label("e");
    Value d = e + c;
    d.label("d");
    Value f = Value(-2.0, "f");
    Value L = d * f;  // L = d * f = (a * b + c) * (-2.0) = -2.0 * (a * b + c)
    L.label("L");
    std::cout << L;

    a = Value(2.0, "a");
    b = Value(-3.0, "b");
    c = Value(10.0 + h, "c");
    e = a * b;
    e.label("e");
    d = e + c;
    d.label("d");
    f = Value(-2.0, "f");
    Value L1 = d * f;  // L = d * f = (a * b + c) * (-2.0) = -2.0 * (a * b + c)
    L1.label("L1");

    std::cout << "dL/da = " << (L1.data() - L.data()) / h;
}