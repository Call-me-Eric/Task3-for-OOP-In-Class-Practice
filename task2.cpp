#include "header.h"
#include <fstream>
#include <iostream>
#include <string>
#include <cctype>

void FileStream(std::string Filename) {
    std::ofstream outFile(Filename);
    if (outFile.is_open()) {
        outFile << "Hello World!";
        outFile.close();
        std::cout << "步骤1完成：已写入Hello World!" << std::endl;
    } else {
        std::cout << "文件打开失败！请检查文件名是否有误" << std::endl;
    }

    outFile << "After close"; 
    std::cout << "步骤2完成：关闭后无法写入，文件内容仍为Hello World!" << std::endl;

    outFile.open(Filename);
    if (outFile.is_open()) {
        outFile << "test";
        outFile.close();
        std::cout << "步骤3完成：已用test覆盖原有内容" << std::endl;
    }

    outFile.open(Filename, std::ios::app);
    if (outFile.is_open()) {
        outFile << "It's open again";
        outFile.close();
        std::cout << "步骤4完成：已追加内容It's open again" << std::endl;
    }
}

int countWords(std::string Filename,std::string pattern)
{
    std::ifstream inFile(Filename);
    if (!inFile.is_open()) {
        return -1;
    }

    int count = 0;
    std::string word;
    
    for(int i=0; i<pattern.size(); i++){
        pattern[i] = tolower(pattern[i]);
    }

    while (inFile >> word) {
        std::string temp;

        for(int i=0; i<word.size(); i++){
            char c = word[i];
            if(isalpha(c)){
                temp += tolower(c);
            }
        }

        if(temp == pattern){
            count++;
        }
    }
    
    inFile.close();
    return count;
}