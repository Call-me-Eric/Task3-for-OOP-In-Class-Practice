#include<iostream>
#include<cstring>
using namespace std;

class Person{
private:
    char* name;
    int sex;
    int age;
public:
    Person(){
        name=nullptr;
        sex=0;
        age=0;
    }

    Person(const char* n,int s,int a){
        if(n!=nullptr){
            name=new char[strlen(n)+1];
            strcpy_s(name,strlen(n)+1,n);
        }
        else{
            name=nullptr;
        }
        sex=s;
        age=a;
    }

    Person(const Person &c){
        if(c.name!=nullptr){
            name=new char[strlen(c.name)+1];
            strcpy_s(name,strlen(c.name)+1,c.name);
        }
        else{
            name=nullptr;
        }
        sex=c.sex;
        age=c.age;
    }

    void Set(const char* n,int s,int a){
        delete[] name;
        char* p=new char[strlen(n)+1];
        name=p;
        strcpy_s(name,strlen(n)+1,n);
        sex=s;
        age=a;
    }

    void print(){
        cout<<"name:"<<name<<endl;
        cout<<"sex:";
        if(sex==0){
            cout<<"male"<<endl;
        }
        else{
            cout<<"female"<<endl;
        }
        cout<<"age:"<<age<<endl;
    }

    ~Person(){
        delete[] name;
    }
};

int main(){
    Person p1;
    p1.Set("Rose",1,18);
    p1.print();

    Person p2("Jack",0,20);
    p2.print();

    Person p3=p2;
    p3.print();

    return 0;
}