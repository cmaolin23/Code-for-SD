// diversity_exact_gcc8.cpp
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

using namespace std;

// 并查集维护连通片
struct DSU
{
    vector<int> p, r;
    DSU(int n = 0) { init(n); }
    void init(int n)
    {
        p.resize(n);
        r.assign(n, 0);
        iota(p.begin(), p.end(), 0);
    }
    int find(int x) { return p[x] == x ? x : p[x] = find(p[x]); }
    bool unite(int a, int b)
    {
        a = find(a);
        b = find(b);
        if (a == b)
            return false;
        if (r[a] < r[b])
            swap(a, b);
        p[b] = a;
        if (r[a] == r[b])
            r[a]++;
        return true;
    }
};

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
    string test_file;
};

static void parse_args(int argc, char **argv, Params &P)
{
    if (argc < 4)
    {
        cerr << "Usage: " << argv[0] << " <graph.txt> <tau> <b> [--trials T] [--seed S] [--query RAW_ID] [--directed]\n";
        exit(1);
    }
    P.path = argv[1];
    P.tau = stoi(argv[2]);
    P.b = stoi(argv[3]);
    for (int i = 4; i < argc; i++)
    {
        string s = argv[i];
        if (s == "--trials" && i + 1 < argc)
        {
            P.trials = stoi(argv[++i]);
        }
        else if (s == "--seed" && i + 1 < argc)
        {
            P.seed = stoull(argv[++i]);
        }
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
        {
            P.directed = true;
        }
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

struct Graph
{
    vector<long long> id2raw;
    unordered_map<long long, int> raw2id;
    vector<vector<int>> adj;
};

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
        G.raw2id[all[i]] = i;
    G.adj.assign(all.size(), {});

    // 建图（无向）
    for (auto &e : edges)
    {
        int a = G.raw2id[e.first], b = G.raw2id[e.second];
        G.adj[a].push_back(b);
        G.adj[b].push_back(a);
    }
    return G;
}

static void build_neighbor_subgraph(const Graph &G, int u,
                                    vector<int> &neighbors,
                                    vector<vector<int>> &subAdj,
                                    vector<int> &map_to_sub,
                                    vector<int> &sub_to_global)
{
    const auto &adj = G.adj;
    vector<char> isN(G.adj.size(), 0);
    for (int v : adj[u])
    {
        isN[v] = 1;
        neighbors.push_back(v);
    }
    sort(neighbors.begin(), neighbors.end());
    neighbors.erase(unique(neighbors.begin(), neighbors.end()), neighbors.end());

    int m = neighbors.size();
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
    {
        if (vis[i])
            continue;
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
        comps.push_back(comp);
    }
    return comps;
}

struct BnBResult
{
    int best_q = 0;
    vector<vector<int>> best_groups;
};

static void bnb_dfs(int idx, int k, int i_groups, int tau,
                    const vector<int> &sizes,
                    vector<int> &load,
                    int q, int Srem,
                    BnBResult &res,
                    vector<vector<int>> &cur_groups)
{
    int D = 0;
    for (int g = 0; g < i_groups; ++g)
        D += max(0, tau - load[g]);
    int UB = q + min(i_groups - q, (Srem + D) / tau);
    if (UB <= res.best_q)
        return;

    int empty_groups = 0;
    for (int g = 0; g < i_groups; ++g)
        if (cur_groups[g].empty())
            empty_groups++;
    if (k - idx < empty_groups)
        return;

    if (idx == k)
    {
        if (q > res.best_q)
        {
            res.best_q = q;
            res.best_groups = cur_groups;
        }
        return;
    }

    for (int g = 0; g < i_groups; ++g)
    {
        int need_before = max(0, tau - load[g]);
        load[g] += sizes[idx];
        int need_after = max(0, tau - load[g]);
        int delta = (need_before > 0 && need_after == 0) ? 1 : 0;

        cur_groups[g].push_back(idx);
        bnb_dfs(idx + 1, k, i_groups, tau, sizes, load, q + delta, Srem - sizes[idx], res, cur_groups);
        cur_groups[g].pop_back();
        load[g] -= sizes[idx];
    }
}

static BnBResult branch_and_bound(const vector<int> &sizes, int k, int i_groups, int tau)
{
    BnBResult res;
    vector<int> load(i_groups, 0);
    int Srem = 0;
    for (int t = 0; t < k; ++t)
        Srem += sizes[t];
    vector<vector<int>> cur_groups(i_groups);
    bnb_dfs(0, k, i_groups, tau, sizes, load, 0, Srem, res, cur_groups);
    return res;
}

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

//         int q0 = 0;
//         struct CompInfo
//         {
//             int size;
//             vector<int> nodes;
//         };
//         vector<CompInfo> U;
//         U.reserve(comps.size());
//         for (auto &comp : comps)
//         {
//             if ((int)comp.size() >= P.tau)
//                 q0++;
//             else
//             {
//                 CompInfo ci;
//                 ci.size = (int)comp.size();
//                 for (int x : comp)
//                     ci.nodes.push_back(sub_to_global[x]);
//                 U.push_back(ci);
//             }
//         }

//         vector<pair<int, int>> added_edges;
//         int best_increment = 0;
//         if (!U.empty() && P.b > 0 && P.tau > 0)
//         {
//             sort(U.begin(), U.end(), [](const CompInfo &a, const CompInfo &b)
//                  { return a.size > b.size; });
//             vector<int> sizes;
//             for (auto &ci : U)
//                 sizes.push_back(ci.size);
//             int Ucnt = (int)U.size();
//             int global_best_q = 0;
//             vector<vector<int>> global_best_groups;

//             int i_max = min(P.b, Ucnt);
//             for (int i_groups = 1; i_groups <= i_max; ++i_groups)
//             {
//                 int k = min(P.b + i_groups, Ucnt);
//                 if (k <= 0 || i_groups <= 0 || i_groups > k)
//                     continue;
//                 BnBResult res = branch_and_bound(sizes, k, i_groups, P.tau);
//                 if (res.best_q > global_best_q)
//                 {
//                     global_best_q = res.best_q;
//                     global_best_groups = res.best_groups;
//                 }
//             }
//             best_increment = global_best_q;

//             // 输出 global_best_groups 信息
//             // if (!global_best_groups.empty())
//             // {
//             //     cout << "global_best_groups (" << global_best_groups.size() << " groups):" << endl;
//             //     for (size_t i = 0; i < global_best_groups.size(); ++i)
//             //     {
//             //         cout << "  Group " << i << ": ";
//             //         for (int comp_idx : global_best_groups[i])
//             //         {
//             //             cout << comp_idx << " "; // 连通分量编号
//             //         }
//             //         cout << " | nodes: ";
//             //         for (int comp_idx : global_best_groups[i])
//             //         {
//             //             for (int node : U[comp_idx].nodes)
//             //             {
//             //                 cout << G.id2raw[node] << " ";
//             //             }
//             //         }
//             //         cout << endl;
//             //     }
//             // }

//             for (const auto &group : global_best_groups)
//             {
//                 if (group.empty())
//                     continue;
//                 vector<int> reps;
//                 for (int comp_idx : group)
//                     if (!U[comp_idx].nodes.empty())
//                         reps.push_back(U[comp_idx].nodes[0]);
//                 for (size_t i = 1; i < reps.size(); ++i)
//                     added_edges.emplace_back(reps[i - 1], reps[i]);
//             }
//         }

//         auto t1 = chrono::high_resolution_clock::now();
//         auto duration_ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
//         double ms = static_cast<double>(duration_ms);

//         int q_new = q0 + best_increment;

//         cout << "=== Trial " << tcase << " ===\n";
//         cout << "Query node (raw): " << G.id2raw[u] << "  (compressed: " << u << ")\n";
//         // // 输出邻接子图的连通分量信息
//         // cout << "Neighbor subgraph connected components (" << comps.size() << "):" << endl;
//         // for (size_t i = 0; i < comps.size(); ++i)
//         // {
//         //     cout << "  Component " << i << " (size " << comps[i].size() << "): ";
//         //     for (int idx : comps[i])
//         //     {
//         //         cout << G.id2raw[sub_to_global[idx]] << " ";
//         //     }
//         //     cout << endl;
//         // }
//         cout << "Neighbors: " << neighbors.size() << ", Components: " << comps.size() << "\n";
//         cout << "Tau: " << P.tau << ", Budget b: " << P.b << "\n";
//         cout << "Initial qualified components q0: " << q0 << "\n";
//         cout << "New qualified after adding edges: " << q_new << "  (increase: " << (q_new - q0) << ")\n";
//         cout << "Time (ms): " << fixed << setprecision(3) << ms << "\n";
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

    // 读取查询节点
    vector<long long> queries;
    ifstream fin(P.test_file);
    if (!fin)
    {
        cerr << "Failed to open test file: " << P.test_file << "\n";
        return 0;
    }
    string line;
    long long q_raw;
    while (getline(fin, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        stringstream ss(line);
        if (ss >> q_raw)
            queries.push_back(q_raw);
    }
    fin.close();
    if (queries.empty())
    {
        cerr << "No queries found in file.\n";
        return 0;
    }

    ofstream fout("result_exact.txt");
    double total_time_ms = 0;
    int total_inc = 0;

    for (size_t tcase = 0; tcase < queries.size(); ++tcase)
    {
        long long raw_q = queries[tcase];
        auto it = G.raw2id.find(raw_q);
        if (it == G.raw2id.end())
        {
            fout << raw_q << "\tERROR_NOT_IN_GRAPH\n";
            continue;
        }
        int u = it->second;

        vector<int> neighbors, map_to_sub, sub_to_global;
        vector<vector<int>> subAdj;
        build_neighbor_subgraph(G, u, neighbors, subAdj, map_to_sub, sub_to_global);

        auto t0 = chrono::high_resolution_clock::now();
        auto comps = connected_components(subAdj);

        int q0 = 0;
        struct CompInfo
        {
            int size;
            vector<int> nodes;
        };
        vector<CompInfo> U;
        for (auto &comp : comps)
        {
            if ((int)comp.size() >= P.tau)
                q0++;
            else
            {
                CompInfo ci;
                ci.size = (int)comp.size();
                for (int x : comp)
                    ci.nodes.push_back(sub_to_global[x]);
                U.push_back(ci);
            }
        }

        vector<pair<int, int>> added_edges;
        int best_increment = 0;
        if (!U.empty() && P.b > 0 && P.tau > 0)
        {
            sort(U.begin(), U.end(), [](const CompInfo &a, const CompInfo &b)
                 { return a.size > b.size; });
            vector<int> sizes;
            for (auto &ci : U)
                sizes.push_back(ci.size);
            int Ucnt = (int)U.size();
            int global_best_q = 0;
            vector<vector<int>> global_best_groups;

            int i_max = min(P.b, Ucnt);
            for (int i_groups = 1; i_groups <= i_max; ++i_groups)
            {
                int k = min(P.b + i_groups, Ucnt);
                if (k <= 0 || i_groups <= 0 || i_groups > k)
                    continue;
                BnBResult res = branch_and_bound(sizes, k, i_groups, P.tau);
                if (res.best_q > global_best_q)
                {
                    global_best_q = res.best_q;
                    global_best_groups = res.best_groups;
                }
            }
            best_increment = global_best_q;

            for (const auto &group : global_best_groups)
            {
                if (group.empty())
                    continue;
                vector<int> reps;
                for (int comp_idx : group)
                    if (!U[comp_idx].nodes.empty())
                        reps.push_back(U[comp_idx].nodes[0]);
                for (size_t i = 1; i < reps.size(); ++i)
                    added_edges.emplace_back(reps[i - 1], reps[i]);
            }
        }

        auto t1 = chrono::high_resolution_clock::now();
        auto duration_ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
        total_time_ms += duration_ms;

        int q_new = q0 + best_increment;
        total_inc += best_increment;

        // 写入结果文件
        fout << raw_q << "\t" << q0 << "\t" << q_new << "\t" << best_increment << "\t" << added_edges.size();
        for (auto &e : added_edges)
            fout << "\t" << G.id2raw[e.first] << "-" << G.id2raw[e.second];
        fout << "\n";

        // 可选打印信息
        cout << "Query node: " << raw_q << ", q0: " << q0 << ", q_new: " << q_new
             << ", inc: " << best_increment << ", edges added: " << added_edges.size()
             << ", time(ms): " << duration_ms << "\n";
    }

    fout << "#TOTAL_INC\t" << total_inc << "\n";
    fout << "#TOTAL_TIME_MS\t" << total_time_ms << "\n";
    fout.close();

    cout << "Exact evaluation done. Total increment: " << total_inc
         << ", Total time(ms): " << total_time_ms << "\n";
    return 0;
}
