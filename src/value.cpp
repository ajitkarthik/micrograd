#include "value.hpp"

#include <cstddef>
#include <iostream>
#include <memory>
#include <numbers>
#include <ostream>
#include <string>

void Value::Node::print(std::ostream& os, int depth,
                        std::unordered_set<const Node*>* visited) const {
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
void Value::Node::backward(void) {
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

// non-leaf (operation) constructor
Value::Value(double data, std::vector<std::shared_ptr<Node>> parents, Node::Op op,
             const std::string& opstring, const std::string& label) {
    node_ = std::make_shared<Node>(data, parents, op, opstring, label);
}

// ===getters===
std::shared_ptr<Value::Node> Value::node() const { return node_; }

// ===setters===
void Value::label(const std::string& label) { node_->label = label; }
void Value::grad(double grad) { node_->grad = grad; }

// ===Constructors===
// leaf node constuctor
Value::Value(double data, const std::string& label) {
    // for leaf value node, initialize with empty parents
    node_ = std::make_shared<Node>(data, std::vector<std::shared_ptr<Node>>{}, Node::Op::LEAF, "",
                                   label);
}

// constructor to go from Node object to Value object
Value::Value(std::shared_ptr<Node> n) { node_ = std::move(n); }

// ===Operator overloads===
// + operator
Value Value::operator+(const Value& other) const {
    return Value(node_->data + other.node_->data, {node_, other.node_}, Node::Op::ADD, "+");
}

// tanh operator
Value Value::tanh() const {
    using std::pow;
    using std::numbers::e;
    return Value((pow(e, 2 * node_->data) - 1) / (pow(e, 2 * node_->data) + 1), {node_},
                 Node::Op::TANH, "tanh");
}

// * operator
Value Value::operator*(const Value& other) const {
    return Value(node_->data * other.node_->data, {node_, other.node_}, Node::Op::MUL, "*");
}

void Value::backward() { node_->backward(); }

std::ostream& operator<<(std::ostream& os, const Value& v) {
    v.node()->print(os);
    return os;
}