#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <chrono>
#include <unordered_set>

using namespace std;

// -----------------------------
// Graph
// -----------------------------
struct Graph
{
    vector<long long> id2raw;
    unordered_map<long long, int> raw2id;
    vector<vector<int>> adj;
};

struct CompInfo
{
    int id;            // 在 U 中的下标（构造后赋值）
    int size;          // 连通分量大小
    vector<int> nodes; // 该分量中的原图节点（邻居子图里的全局索引）
};

// -----------------------------
// 参数
// -----------------------------
struct Params
{
    string path;
    int tau = 5;
    int b = 8;
    int trials = 1;
    uint64_t seed = 42;
    bool has_query = false;
    long long query_raw = -1;
    bool directed = false;
    int method = 3;   // 1:NF 2:SI 3:ISI
    string test_file; // 新增 test 文件
};

static void parse_args(int argc, char **argv, Params &P)
{
    if (argc < 5)
    {
        cerr << "Usage: " << argv[0] << " <graph.txt> <tau> <b> [--trials T] [--seed S] [--query RAW_ID] [--directed]\n";
        exit(1);
    }
    P.path = argv[1];
    P.tau = stoi(argv[2]);
    P.b = stoi(argv[3]);
    P.method = stoi(argv[4]);
    for (int i = 5; i < argc; i++)
    {
        string s = argv[i];
        if (s == "--trials" && i + 1 < argc)
            P.trials = stoi(argv[++i]);
        else if (s == "--seed" && i + 1 < argc)
            P.seed = stoull(argv[++i]);
        else if (s == "--query" && i + 1 < argc)
        {
            P.has_query = true;
            P.query_raw = stoll(argv[++i]);
        }
        else if (s == "--test" && i + 1 < argc)
        {
            P.test_file = argv[++i];
        }
        else if (s == "--directed")
            P.directed = true;
        else
        {
            cerr << "Unknown arg: " << s << "\n";
            exit(1);
        }
    }
    // 检查 test 文件是否指定
    if (P.test_file.empty())
    {
        cerr << "Warning: --test file not specified. No queries will be read from test.\n";
    }
}

// -----------------------------
// 读图（无向）
// -----------------------------
static Graph read_graph(const string &path, bool directed)
{
    Graph G;
    vector<pair<long long, long long>> edges;
    edges.reserve(1 << 20);

    ifstream fin(path);
    if (!fin)
    {
        cerr << "Failed to open " << path << "\n";
        exit(1);
    }

    string line;
    long long u, v;
    while (getline(fin, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        stringstream ss(line);
        if (!(ss >> u >> v))
            continue;
        edges.emplace_back(u, v);
    }

    // 压缩ID
    vector<long long> all;
    all.reserve(edges.size() * 2);
    for (auto &e : edges)
    {
        all.push_back(e.first);
        all.push_back(e.second);
    }
    sort(all.begin(), all.end());
    all.erase(unique(all.begin(), all.end()), all.end());
    G.id2raw = all;
    G.raw2id.reserve(all.size() * 2);
    for (size_t i = 0; i < all.size(); ++i)
        G.raw2id[all[i]] = (int)i;
    G.adj.assign(all.size(), {});

    // 建无向
    for (auto &e : edges)
    {
        int a = G.raw2id[e.first], b = G.raw2id[e.second];
        G.adj[a].push_back(b);
        G.adj[b].push_back(a);
    }
    return G;
}

// -----------------------------
// 邻居子图 & 连通分量
// -----------------------------
static void build_neighbor_subgraph(const Graph &G, int u,
                                    vector<int> &neighbors,
                                    vector<vector<int>> &subAdj,
                                    vector<int> &map_to_sub,
                                    vector<int> &sub_to_global)
{
    for (int v : G.adj[u])
        neighbors.push_back(v);
    sort(neighbors.begin(), neighbors.end());
    neighbors.erase(unique(neighbors.begin(), neighbors.end()), neighbors.end());

    int m = (int)neighbors.size();
    map_to_sub.assign(G.adj.size(), -1);
    sub_to_global.resize(m);
    for (int i = 0; i < m; i++)
    {
        map_to_sub[neighbors[i]] = i;
        sub_to_global[i] = neighbors[i];
    }
    subAdj.assign(m, {});
    for (int i = 0; i < m; i++)
    {
        int gvi = sub_to_global[i];
        for (int w : G.adj[gvi])
        {
            int j = (w >= 0 && w < (int)map_to_sub.size()) ? map_to_sub[w] : -1;
            if (j != -1)
                subAdj[i].push_back(j);
        }
    }
}

static vector<vector<int>> connected_components(const vector<vector<int>> &subAdj)
{
    int n = (int)subAdj.size();
    vector<int> vis(n, 0);
    vector<vector<int>> comps;
    for (int i = 0; i < n; i++)
        if (!vis[i])
        {
            vector<int> comp;
            queue<int> q;
            q.push(i);
            vis[i] = 1;
            while (!q.empty())
            {
                int x = q.front();
                q.pop();
                comp.push_back(x);
                for (int y : subAdj[x])
                    if (!vis[y])
                    {
                        vis[y] = 1;
                        q.push(y);
                    }
            }
            comps.push_back(move(comp));
        }
    return comps;
}

// -----------------------------
// 贪心策略：生成 bins（每个 bin 是 U 的下标列表）
// 假设输入 U 已按 size 降序
// -----------------------------

// Next Fit（优先最大，顺序消耗）
static vector<vector<int>> nextFitBins(const vector<CompInfo> &U, int tau)
{
    vector<vector<int>> bins;
    if (U.empty())
        return bins;

    int n = (int)U.size();
    int i = 0;
    while (i < n)
    {
        int sum = 0;
        vector<int> bin;
        while (i < n && sum < tau)
        {
            bin.push_back(U[i].id);
            sum += U[i].size;
            ++i;
        }
        if (!bin.empty())
            bins.push_back(move(bin));
    }
    return bins;
}

// Simple：最短前缀 < tau，然后从尾部补到 ≥tau，打包后移除，重复
static vector<vector<int>> simpleBins(const vector<CompInfo> &U, int tau)
{
    vector<vector<int>> bins;
    if (U.empty())
        return bins;

    // 先按 size 降序排序，并保存原始 id
    vector<pair<int, int>> items; // pair<size, id>
    for (int i = 0; i < (int)U.size(); i++)
        items.emplace_back(U[i].size, i);
    sort(items.begin(), items.end(), greater<>()); // size 大的在前

    int l = 0, r = items.size() - 1;

    while (l <= r)
    {
        int sum = 0;
        vector<int> bin;

        // 从前端累加
        while (l <= r && sum + items[l].first < tau)
        {
            sum += items[l].first;
            bin.push_back(items[l].second);
            l++;
        }

        // 如果还没达到 tau，从尾端补
        while (l <= r && sum < tau)
        {
            sum += items[r].first;
            bin.push_back(items[r].second);
            r--;
        }

        bins.push_back(move(bin));
    }

    return bins;
}

// Improved Simple (ISI)
static vector<vector<int>> improvedSimpleBins(const vector<CompInfo> &U, int tau)
{
    vector<int> X, Y, Z;
    for (int i = 0; i < (int)U.size(); i++)
    {
        if (U[i].size >= tau / 2.0)
            X.push_back(i);
        else if (U[i].size >= tau / 3.0)
            Y.push_back(i);
        else
            Z.push_back(i);
    }

    // 各组内部按 size 降序排序
    auto cmp = [&](int a, int b)
    { return U[a].size < U[b].size; };
    sort(X.begin(), X.end(), cmp);
    sort(Y.begin(), Y.end(), cmp);
    sort(Z.begin(), Z.end(), cmp);
    // 输出 X, Y, Z 分组
    // cout << "X group (size >= tau/2): ";
    // for (int idx : X)
    //     cout << "(id=" << idx << ",sz=" << U[idx].size << ") ";
    // cout << "\n";
    // cout << "Y group (size >= tau/3): ";
    // for (int idx : Y)
    //     cout << "(id=" << idx << ",sz=" << U[idx].size << ") ";
    // cout << "\n";
    // cout << "Z group (size < tau/3): ";
    // for (int idx : Z)
    //     cout << "(id=" << idx << ",sz=" << U[idx].size << ") ";
    // cout << "\n";

    vector<vector<int>> bins;

    // Phase 1: 用 X 或 2 个 Y + Z 填充 bin，直到 bin sum >= tau
    while ((!X.empty() || Y.size() >= 2) && !Z.empty())
    {
        int sum = 0;
        vector<int> bin;

        // 先加入 X 或两个 Y
        if (!X.empty())
        {
            bin.push_back(X.back());
            sum += U[X.back()].size;
            X.pop_back();
        }
        else if (Y.size() >= 2)
        {
            bin.push_back(Y.back());
            sum += U[Y.back()].size;
            Y.pop_back();

            bin.push_back(Y.back());
            sum += U[Y.back()].size;
            Y.pop_back();
        }

        // 然后逐个加入 Z
        auto it = Z.begin();
        while (sum < tau && it != Z.end())
        {
            bin.push_back(*it);
            sum += U[*it].size;
            it = Z.erase(it); // 每次都删掉已加入的元素
        }

        bins.push_back(move(bin));
        // 输出当前 bins 的所有元素
        // for (size_t bi = 0; bi < bins.size(); ++bi)
        // {
        //     int sum = 0;
        //     for (int idx : bins[bi])
        //         sum += U[idx].size;
        //     cout << "  Bin " << bi << " [sum=" << sum << "]: ";
        //     for (int idx : bins[bi])
        //         cout << "(id=" << idx << ",sz=" << U[idx].size << ") ";
        //     cout << "\n";
        // }
    }

    // Phase 2: 处理剩余 X/Y/Z
    auto nextFit = [&](vector<int> &vec)
    {
        sort(vec.begin(), vec.end(), [&](int a, int b)
             { return U[a].size > U[b].size; });
        int l = 0, r = vec.size() - 1;
        while (l <= r)
        {
            int sum = 0;
            vector<int> bin;
            while (l <= r && sum + U[vec[l]].size < tau)
            {
                sum += U[vec[l]].size;
                bin.push_back(vec[l]);
                l++;
            }
            while (l <= r && sum < tau)
            {
                sum += U[vec[r]].size;
                bin.push_back(vec[r]);
                r--;
            }
            bins.push_back(move(bin));
        }
    };

    nextFit(Z);

    // X 单独处理：尽量两两组合
    while ((int)X.size() >= 2)
    {
        bins.push_back({X.back(), X[X.size() - 2]});
        X.pop_back();
        X.pop_back();
    }
    if (!X.empty())
        bins.push_back({X.back()});

    // Y 单独处理：尽量三三组合
    while ((int)Y.size() >= 3)
    {
        bins.push_back({Y.back(), Y[Y.size() - 2], Y[Y.size() - 3]});
        Y.pop_back();
        Y.pop_back();
        Y.pop_back();
    }
    if (!Y.empty())
        bins.push_back(Y); // 剩余不足三的 Y

    return bins;
}

// -----------------------------
// 依据 bins 选择边（尊重预算 b）并计算多样性增量
// 每个可覆盖 bin（sum>=tau）若 edges_needed = t-1 <= rem_b，就选中并加 (t-1) 条边，贡献 +1
// -----------------------------
static vector<pair<int, int>> planEdgesFromBins(
    const vector<CompInfo> &U,
    const vector<vector<int>> &bins,
    int tau, int b, int &qualified_bins)
{

    vector<pair<int, int>> added_edges;
    int rem = b;
    qualified_bins = 0;

    for (const auto &bin : bins)
    {
        if (bin.empty())
            continue;

        // 计算该 bin 的总和与需要的边数
        // 输出 bins 内容
        cout << "Bins (" << bins.size() << "):\n";
        for (size_t bi = 0; bi < bins.size(); ++bi)
        {
            int sum = 0;
            cout << "  Bin " << bi << " [sum=";
            for (int idx : bins[bi])
                sum += U[idx].size;
            cout << sum << "]: ";
            for (int idx : bins[bi])
                cout << "(id=" << idx << ",sz=" << U[idx].size << ") ";
            cout << "\n";
        }
        int sum = 0;
        for (int idx : bin)
            sum += U[idx].size;
        if (sum < tau)
            continue; // 该 bin 本身不达标，跳过

        int need = (int)bin.size() - 1; // 链接 t 个分量需要 t-1 条边
        if (need <= rem)
        {
            // 以每个分量的一个代表点串起来形成链
            vector<int> reps;
            reps.reserve(bin.size());
            for (int idx : bin)
            {
                if (!U[idx].nodes.empty())
                    reps.push_back(U[idx].nodes[0]);
            }
            for (size_t i = 1; i < reps.size(); ++i)
            {
                added_edges.emplace_back(reps[i - 1], reps[i]);
            }
            rem -= need;
            qualified_bins++;
            if (rem == 0)
                break;
        }
        else
        {
            // 预算不够链接完该 bin，则终止（也可以尝试下一个更小的 bin；这里直接继续看下一个）
            continue;
        }
    }
    return added_edges;
}

// -----------------------------
// 主流程
// -----------------------------
// int main(int argc, char **argv)
// {
//     ios::sync_with_stdio(false);
//     cin.tie(nullptr);

//     Params P;
//     parse_args(argc, argv, P);
//     Graph G = read_graph(P.path, P.directed);
//     const int n = (int)G.adj.size();
//     if (n == 0)
//     {
//         cerr << "Empty graph.\n";
//         return 0;
//     }

//     mt19937_64 rng(P.seed);
//     vector<int> candidates;
//     for (int u = 0; u < n; ++u)
//         if (!G.adj[u].empty())
//             candidates.push_back(u);
//     if (candidates.empty())
//     {
//         cerr << "No node with degree>0.\n";
//         return 0;
//     }

//     for (int tcase = 1; tcase <= P.trials; ++tcase)
//     {
//         int u;
//         if (P.has_query)
//         {
//             auto it = G.raw2id.find(P.query_raw);
//             if (it == G.raw2id.end())
//             {
//                 cerr << "Query raw id not in graph. Pick random.\n";
//                 u = candidates[rng() % candidates.size()];
//             }
//             else
//                 u = it->second;
//         }
//         else
//             u = candidates[rng() % candidates.size()];

//         vector<int> neighbors, map_to_sub, sub_to_global;
//         vector<vector<int>> subAdj;
//         build_neighbor_subgraph(G, u, neighbors, subAdj, map_to_sub, sub_to_global);

//         auto t0 = chrono::high_resolution_clock::now();
//         auto comps = connected_components(subAdj);

//         // 统计 q0，并收集未达阈值的分量 U
//         int q0 = 0;
//         vector<CompInfo> U;
//         U.reserve(comps.size());
//         for (size_t i = 0; i < comps.size(); ++i)
//         {
//             int sz = (int)comps[i].size();
//             if (sz >= P.tau)
//             {
//                 q0++;
//             }
//             else
//             {
//                 CompInfo ci;
//                 ci.size = sz;
//                 ci.nodes.reserve(sz);
//                 for (int x : comps[i])
//                     ci.nodes.push_back(sub_to_global[x]);
//                 U.push_back(move(ci));
//             }
//         }

//         // 给 U 排序（降序）并赋 id
//         sort(U.begin(), U.end(), [](const CompInfo &a, const CompInfo &b)
//              { return a.size > b.size; });
//         for (int i = 0; i < (int)U.size(); ++i)
//             U[i].id = i;

//         // 选择贪心策略：NF / SI / ISI
//         int method = P.method;
//         vector<vector<int>> bins;
//         if (method == 1)
//             bins = nextFitBins(U, P.tau);
//         else if (method == 2)
//             bins = simpleBins(U, P.tau);
//         else
//             bins = improvedSimpleBins(U, P.tau);

//         // 依据 bins 在预算内选边，并计算新覆盖的 bin 数
//         int covered_bins = 0;
//         vector<pair<int, int>> added_edges = planEdgesFromBins(U, bins, P.tau, P.b, covered_bins);

//         auto t1 = chrono::high_resolution_clock::now();
//         double ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();

//         int q_new = q0 + covered_bins;

//         // ------- 输出（对齐 exact 风格） -------
//         cout << "=== Trial " << tcase << " ===\n";
//         cout << "Query node (raw): " << G.id2raw[u] << "  (compressed: " << u << ")\n";

//         cout << "Neighbor subgraph connected components (" << comps.size() << "):\n";
//         for (size_t i = 0; i < comps.size(); ++i)
//         {
//             cout << "  Component " << i << " (size " << comps[i].size() << "): ";
//             for (int idx : comps[i])
//                 cout << G.id2raw[sub_to_global[idx]] << " ";
//             cout << "\n";
//         }

//         cout << "Neighbors: " << neighbors.size() << ", Components: " << comps.size() << "\n";
//         cout << "Tau: " << P.tau << ", Budget b: " << P.b << "\n";
//         cout << "Initial qualified components q0: " << q0 << "\n";
//         cout << "New qualified after adding edges: " << q_new
//              << "  (increase: " << (q_new - q0) << ")\n";
//         cout << "Time (ms): " << fixed << setprecision(3) << ms << "\n";

//         // 打印贪心 bins 摘要（可注释掉）
//         cout << "Greedy bins (" << bins.size() << "):\n";
//         for (size_t bi = 0; bi < bins.size(); ++bi)
//         {
//             int sum = 0;
//             for (int idx : bins[bi])
//                 sum += U[idx].size;
//             cout << "  Bin " << bi << " [sum=" << sum << "]: ";
//             for (int idx : bins[bi])
//                 cout << "(id=" << idx << ",sz=" << U[idx].size << ") ";
//             cout << "\n";
//         }

//         cout << "Edges to add (" << added_edges.size() << "):\n";
//         for (auto &e : added_edges)
//             cout << G.id2raw[e.first] << " " << G.id2raw[e.second] << "\n";
//         cout << flush;
//     }
//     return 0;
// }

int main(int argc, char **argv)
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Params P;
    parse_args(argc, argv, P);
    Graph G = read_graph(P.path, P.directed);
    const int n = (int)G.adj.size();
    if (n == 0)
    {
        cerr << "Empty graph.\n";
        return 0;
    }

    // 读取 test 查询节点
    vector<long long> test_queries_raw;
    ifstream fin(P.test_file);
    if (!fin)
    {
        cerr << "Cannot open test file.\n";
        return 1;
    }
    long long qid;
    while (fin >> qid)
        test_queries_raw.push_back(qid);

    mt19937_64 rng(P.seed);

    double total_ms = 0.0;
    int total_diversity_increase = 0;

    ofstream fout("results.txt");
    if (!fout)
    {
        cerr << "Cannot open output file.\n";
        return 1;
    }

    int trial = 0;
    for (long long raw_u : test_queries_raw)
    {
        trial++;
        auto it = G.raw2id.find(raw_u);
        int u;
        if (it == G.raw2id.end())
        {
            cerr << "Query raw id " << raw_u << " not in graph. Pick random.\n";
            u = rng() % n;
        }
        else
            u = it->second;

        vector<int> neighbors, map_to_sub, sub_to_global;
        vector<vector<int>> subAdj;
        build_neighbor_subgraph(G, u, neighbors, subAdj, map_to_sub, sub_to_global);

        auto t0 = chrono::high_resolution_clock::now();
        auto comps = connected_components(subAdj);

        // 统计 q0，并收集未达阈值的分量 U
        int q0 = 0;
        vector<CompInfo> U;
        for (size_t i = 0; i < comps.size(); ++i)
        {
            int sz = (int)comps[i].size();
            if (sz >= P.tau)
                q0++;
            else
            {
                CompInfo ci;
                ci.size = sz;
                ci.nodes.reserve(sz);
                for (int x : comps[i])
                    ci.nodes.push_back(sub_to_global[x]);
                U.push_back(move(ci));
            }
        }

        // 给 U 排序并赋 id
        sort(U.begin(), U.end(), [](const CompInfo &a, const CompInfo &b)
             { return a.size > b.size; });
        for (int i = 0; i < (int)U.size(); ++i)
            U[i].id = i;

        vector<vector<int>> bins = improvedSimpleBins(U, P.tau);

        int covered_bins = 0;
        vector<pair<int, int>> added_edges = planEdgesFromBins(U, bins, P.tau, P.b, covered_bins);

        auto t1 = chrono::high_resolution_clock::now();
        double ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();

        total_ms += ms;
        total_diversity_increase += covered_bins;

        fout << "Query " << raw_u
             << " q0=" << q0
             << " new_q=" << (q0 + covered_bins)
             << " increase=" << covered_bins
             << " time_ms=" << ms << "\n";
    }

    fout << "\nTotal diversity increase: " << total_diversity_increase << "\n";
    fout << "Total time (ms): " << total_ms << "\n";
    fout.close();

    cout << "Done. Summary written to " << "results.txt" << "\n";
    return 0;
}

// g++ -std=c++14 -O2 -Wall -Wextra -o greedy greedy.cpp
//./greedy ../data/web-Google.txt 3 4 1 --trials 1 --query 22115
//./greedy ../data/web-Google.txt 5 5 1 --test ../data/test.txt --seed 42 --trials 1