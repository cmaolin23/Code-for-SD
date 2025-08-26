#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <random>

using namespace std;

// 读取边列表文件，转为无向图
unordered_map<int, unordered_set<int>> readGraph(const string &filename)
{
    ifstream fin(filename);
    if (!fin)
    {
        cerr << "err " << filename << endl;
        exit(1);
    }

    unordered_map<int, unordered_set<int>> graph;
    int u, v;
    while (fin >> u >> v)
    {
        graph[u].insert(v);
        graph[v].insert(u); // 转为无向图
    }
    return graph;
}

// 计算邻居子图的连通分量数量
int countConnectedComponents(const unordered_map<int, unordered_set<int>> &graph, int q)
{
    const auto &neighbors = graph.at(q);
    unordered_set<int> visited;
    int components = 0;

    for (int u : neighbors)
    {
        if (visited.count(u))
            continue;
        components++;
        queue<int> qQueue;
        qQueue.push(u);
        visited.insert(u);

        while (!qQueue.empty())
        {
            int node = qQueue.front();
            qQueue.pop();
            for (int v : graph.at(node))
            {
                if (neighbors.count(v) && !visited.count(v))
                {
                    visited.insert(v);
                    qQueue.push(v);
                }
            }
        }
    }
    return components;
}

// 划分训练集和测试集
void splitDataset(const vector<int> &nodes, double trainRatio,
                  vector<int> &trainSet, vector<int> &testSet)
{
    vector<int> shuffled = nodes;
    random_device rd;
    mt19937 g(rd());
    shuffle(shuffled.begin(), shuffled.end(), g);

    size_t trainSize = static_cast<size_t>(shuffled.size() * trainRatio);
    trainSet.assign(shuffled.begin(), shuffled.begin() + trainSize);
    testSet.assign(shuffled.begin() + trainSize, shuffled.end());
}

void saveToFile(const string &filename, const vector<int> &nodes)
{
    ofstream fout(filename);
    for (int node : nodes)
    {
        fout << node << "\n";
    }
}

int main()
{
    string inputFile = "web-Google.txt"; // 输入图文件
    int threshold = 20;                  // 连通分量数量阈值
    double trainRatio = 0.8;             // 训练集比例

    auto graph = readGraph(inputFile);
    vector<int> selectedNodes;

    for (const auto &pair : graph)
    {
        int q = pair.first;
        const auto &neighbors = graph[q];
        if (neighbors.size() < threshold)
            continue;
        int components = countConnectedComponents(graph, q);
        // if (components >= threshold)
        if (components <= 20 && components >= 15)
        {
            selectedNodes.push_back(q);
        }
    }

    cout << "select " << selectedNodes.size() << " nodes" << endl;

    vector<int> trainSet, testSet;
    splitDataset(selectedNodes, trainRatio, trainSet, testSet);

    saveToFile("train.txt", trainSet);
    saveToFile("test.txt", testSet);

    return 0;
}
