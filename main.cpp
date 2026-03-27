#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include "header.h"
using std::cin, std::cout;
   
void cleanscreen()
{
    #if defined(_WIN32) || defined(_WIN64)
    system("cls");
    #else
    system("clear");
    #endif
}
int main()
{
    while(1)
    {
        cleanscreen();
        cout << "请输入你想进行的操作指令.\n";
        cout << "1:调用classExample.\n";
        cout << "2:调用extractedQuote.\n";
        cout << "3:调用FileStream.\n";
        cout << "4:调用countWords.\n";
        cout << "键入exit终止程序\n";
        std::string op;
        cin >> op;
        if(op == "1")
        {
            cout << "请输入需要提取整数的字符串\n";
            std::string str;
            cout << "所求结果为:" << classExample(str) << '\n';
        }
        else if(op == "2")
        {
            cout << "请输入一行句子\n";
            extractedQuote();
        }
        else if(op == "3")
        {
            cout << "请输入文件名(包含后缀名)\n";
            std::string FileName;
            cin >> FileName;
            FileStream(FileName);
        }
        else if(op == "4")
        {
            cout << "请输入文件名(包括后缀名)\n";
            std::string FileName;
            cin >> FileName;
            cout << "请输入需要匹配的单词\n";
            std::string pattern;
            cin >> pattern;
            cout << "匹配次数为:" << countWords(FileName,pattern) << '\n';
        }
        else if(op == "exit")
        {
            break;
        }
        else 
        {
            cout << "输入了错误的指令，请重新输入\n";
        }
        cout << "键入回车以继续\n";cin >> op;
    }
    return 0;
}
//设计交互式菜单，可选择功能