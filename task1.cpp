#include<string>
#include<cctype>
#include<sstream>
#include<iostream>
using namespace std;

int classExample(string str){
    int sum=0;
    for(int i=0;i<str.size();i++){
        if(isdigit(str[i])){
            sum+=str[i]-'0';
        }
    }
    return sum;
}

void extractedQuote(){
    string first_name,last_name;
    cin>>first_name>>last_name;
    string quote,word;
    while(cin>>word){
        quote=quote+word+" ";
    }
    cout<<first_name<<" "<<last_name<<" said this: "<<quote<<endl;

    //string sentence;
    //getline(cin,sentence);
    //stringstream ss(sentence);
    //string first_name,last_name,quote;
    //ss>>first_name>>last_name;
    //getline(ss,quote);
    //cout<<first_name<<" "<<last_name<<" said this:"<<quote<<endl;

}
