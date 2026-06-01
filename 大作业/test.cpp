#include<cstring>
#include<cstdio>
#include<algorithm>
#include<iostream>
#include<fstream>
#include<vector>
using std::cerr, std::cout, std::cin;

int main()
{
    int total = 0, tot_suc = 0;
    std::vector<std::pair<std::string, int>> unsuccess_test;
    std::vector<int> succ(10, 0);
    for(int x = 0;x <= 9;x++)
    {
        int cnt = 0, success = 0;
        for(int y = 0;;y++)
        {
            std::string filepath = "tests/mnist_" + std::to_string(x) + "_" + std::to_string(y) + ".raw";
            std::ifstream file(filepath);
            if(!file.is_open())
            {
                break;
            }
            cnt++;
            file.close();
            system(("./A " + filepath + " weights 0 > ans.txt").c_str());
            std::ifstream resultfile("ans.txt");
            int n;resultfile >> n;
            if(n == x)success++,succ[x]++;
            else unsuccess_test.emplace_back(filepath, n);
            // cerr << "test " << filepath << " gives ans = " << n << std::endl;
            resultfile.close();
            // getchar();
        }
        cout << "tested " << cnt << " tests" << std::endl;
        cout << success << " tests are right" << std::endl;
        total += cnt;
        tot_suc += success;
    }
    for(int x = 0;x < 1000;x++)
    {
        int cnt = 0, success = 0;
        for(int y = 0;y < 10;y++)
        {
            std::string filepath = "tests/test_" + std::string(4 - std::to_string(x).length(),'0') + std::to_string(x) + "_label_" + std::to_string(y) + ".raw";
            std::ifstream file(filepath);
            if(!file.is_open())
            {
                continue;
            }
            cnt++;
            file.close();
            system(("./A " + filepath + " weights 0 > ans.txt").c_str());
            std::ifstream resultfile("ans.txt");
            int n;resultfile >> n;
            if(n == y)success++,succ[y]++;
            else unsuccess_test.emplace_back(filepath, n);
            // cerr << "test " << filepath << " gives ans = " << n << std::endl;
            resultfile.close();
            // getchar();
        }
        total += cnt;
        tot_suc += success;
    }
    cout << "total = " << total << " and success = " << tot_suc << std::endl;
    cout << "unsuccessful tests are : " << std::endl;
    for(auto i : unsuccess_test)cout << i.first << " but output ans = " << i.second << std::endl;
}