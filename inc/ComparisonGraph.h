#ifndef COMPARISONGRAPH_HEADER
#define COMPARISONGRAPH_HEADER
        
#include <unordered_set>
#include <unordered_map>
#include <utility>
        
        class ComparisonGraph {
        protected:
            int m; //number of edges
            std::unordered_set<int> node_set;
            std::unordered_map<int, std::unordered_map<int, std::pair<int, int>>> BN;  // Maps nodes to edges with (B, N) weights
        public:
            ComparisonGraph();
            int number_of_nodes() const;
            int number_of_edges() const;
            std::unordered_set<int> nodes() const;
            std::unordered_map<int, std::pair<int, int>> neighbors(int) const;  // Return neighbours with B, N weights
            int degree(int) const;
            bool has_node(int) const;
            bool has_edge(int, int) const;
            void add_node(int);
            void remove_node(int);
            void add_edge(int, int, int B, int N);  // Add weighted edge
            void remove_edge(int, int);
            std::pair<int, int> get_edge_weights(int, int) const;  // Return B and N for an edge

        };
    
    // Initialise graph with no edges
    ComparisonGraph::ComparisonGraph() : m(0) {}
    
    // Implementing functions
    int ComparisonGraph::number_of_nodes() const { return node_set.size(); }
    int ComparisonGraph::number_of_edges() const { return m; }
    std::unordered_set<int> ComparisonGraph::nodes() const { return node_set; }
    
    std::unordered_map<int, std::pair<int, int>> ComparisonGraph::neighbors(int i) const {
        return BN.at(i);
    }
    
    int ComparisonGraph::degree(int i) const {
        return BN.at(i).size();
    }
    
    bool ComparisonGraph::has_node(int i) const {
        return node_set.count(i);
    }
    
    bool ComparisonGraph::has_edge(int i, int j) const {
        if (has_node(i) && BN.at(i).count(j))
            return true;
        else
            return false;
    }
    
    // Add nodes and edges
    void ComparisonGraph::add_node(int i) {
        if (!has_node(i)) {
            node_set.insert(i);
            BN[i];
        }
    }
    
    void ComparisonGraph::add_edge(int i, int j, int B, int N) {
        if (!has_edge(i, j)) {
            add_node(i);
            add_node(j);
            BN[i][j] = std::make_pair(B, N);  // Store the B, N weights
            BN[j][i] = std::make_pair(N - B, N);  // Reverse weights for j beating i
            m += 1;
        }
    }
    
    // Get the B and N values for an edge
    std::pair<int, int> ComparisonGraph::get_edge_weights(int i, int j) const {
        return BN.at(i).at(j);
    }
    
    // Remove edge
    void ComparisonGraph::remove_edge(int i, int j) {
        if (has_edge(i, j)) {
            BN[i].erase(j);
            BN[j].erase(i);
            m -= 1;
        }
    }
    
#endif

