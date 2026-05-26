#include "value.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <numbers>
#include <ostream>
#include <stack>
#include <string>
#include <unordered_set>

void Value::Node::printGraph(std::ostream& os, int depth,
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
        p->printGraph(os, depth + 1, visited);
    }
}

void Value::printGraph(std::ostream& os) { node_->printGraph(os); }

// void Value::Node::printNode(std::ostream& os) const {
//     os << "Value(";
//     if (!label.empty()) os << label << " : ";
//     os << "data=" << data;
//     if (!opstring.empty()) os << ", opstring=" << opstring;
//     os << ", op=" << static_cast<int>(op);
//     os << ", grad=" << grad;
//     os << ")\n";
// }

// gradient computation - See Karpathy's micrograd video for an explanation of this.
// Topological sort algo.
// IMPLEMENTATION SIMPLIFIED FOR MLPS WITH ONLY ONE OUTPUT NEURON (DAG WITH ONE ROOT)
// TopologicalSortDFS(Graph G):
//     visited = array of size V, initialized to false
//     stack = empty stacko9

//     for each vertex v in G:
//         if visited[v] == false:
//             DFSVisit(v, visited, stack)

//     # The topological order is obtained by popping the stack
//     return stack

// DFSVisit(v, visited, stack):
//     visited[v] = true
//     for each neighbor u of v:
//         if visited[u] == false:
//             DFSVisit(u, visited, stack)
//     stack.push(v)

void Value::Node::DFSVisit(Node* curr, std::unordered_set<Node*>& visited,
                           std::stack<Node*>& stack) const {
    visited.insert(curr);
    for (const auto& p : curr->prev) {
        if (!visited.contains(p.get())) {
            DFSVisit(p.get(), visited, stack);
        }
    }
    stack.push(curr);
}

void Value::Node::backward_local(void) {
    switch (op) {
        case Node::Op::TANH:
            // d(tan x)/dx = (1 - tan(x)^2)
            prev[0]->grad += (1.0 - std::pow(data, 2)) * grad;
            break;
        case Node::Op::ADD:
            prev[0]->grad += 1.0 * grad;
            prev[1]->grad += 1.0 * grad;
            break;
        case Node::Op::MUL:
            prev[0]->grad += prev[1]->data * grad;
            prev[1]->grad += prev[0]->data * grad;
            break;
        case Node::Op::POW: {
            // d(x^n)/dx = n * (x^(n-1))
            // d(x^n)/dn = x^n * ln(x)
            const double base = prev[0]->data;
            const double exp = prev[1]->data;
            prev[0]->grad += (exp * std::pow(base, exp - 1)) * grad;
            if (base > 0.0) {
                prev[1]->grad += std::pow(base, exp) * std::log(base) * grad;
            }
            break;
        }
        case Node::Op::LEAF:
            break;
    }
}

void Value::Node::backward(void) {
    std::stack<Node*> stack;
    std::unordered_set<Node*> visited;

    // Initialize root of toposort with the current node
    Value::Node::DFSVisit(this, visited, stack);

    while (!stack.empty()) {
        Node* top = stack.top();
        top->backward_local();
        stack.pop();
    }
}

// ===getters===
std::shared_ptr<Value::Node> Value::node() const { return node_; }

// ===setters===
void Value::label(const std::string& label) { node_->label = label; }
void Value::grad(double grad) { node_->grad = grad; }

// ===Constructors===
// default constructor
Value::Value() : Value(0.0, "") {}

// leaf node constuctor
Value::Value(double data, const std::string& label) {
    // for leaf value node, initialize with empty parents
    node_ = std::make_shared<Node>(data, std::vector<std::shared_ptr<Node>>{}, Node::Op::LEAF, "",
                                   label);
}

// non-leaf (operation) constructor
Value::Value(double data, std::vector<std::shared_ptr<Node>> parents, Node::Op op,
             const std::string& opstring, const std::string& label) {
    node_ = std::make_shared<Node>(data, parents, op, opstring, label);
}

// constructor to go from Node object to Value object
Value::Value(std::shared_ptr<Node> n) { node_ = std::move(n); }

// ===Operator overloads===
// + operator
Value Value::operator+(const Value& other) const {
    return Value(node_->data + other.node_->data, {node_, other.node_}, Node::Op::ADD, "+");
}

// - operator
Value Value::operator-(const Value& other) const { return *this + (-1.0 * other); }

// tanh operator
Value Value::tanh() const {
    using std::pow;
    using std::numbers::e;
    return Value((pow(e, 2 * node_->data) - 1) / (pow(e, 2 * node_->data) + 1), {node_},
                 Node::Op::TANH, "tanh");
}

Value Value::pow(const Value& exponent) const {
    return Value(std::pow(node_->data, exponent.node_->data), {node_, exponent.node_},
                 Value::Node::Op::POW, "pow");
}

// * operator
Value Value::operator*(const Value& other) const {
    return Value(node_->data * other.node_->data, {node_, other.node_}, Node::Op::MUL, "*");
}

Value operator*(double lhs, const Value& rhs) { return Value(lhs) * rhs; }
Value operator*(const Value& lhs, double rhs) { return lhs * Value(rhs); }
Value operator+(double lhs, const Value& rhs) { return Value(lhs) + rhs; }
Value operator+(const Value& lhs, double rhs) { return lhs + Value(rhs); }
Value operator-(double lhs, const Value& rhs) { return Value(lhs) - rhs; }
Value operator-(const Value& lhs, double rhs) { return lhs - Value(rhs); }

void Value::backward() { node_->backward(); }

std::ostream& operator<<(std::ostream& os, const Value& v) {
    os << "Value(" << v.node_->data << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<Value>& v) {
    os << '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ", ";
        os << v[i];
    }
    return os << ']';
}