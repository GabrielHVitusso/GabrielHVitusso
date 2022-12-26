#include <iostream>

using namespace std;

class NPC {
//Variaveis
string _Nome;
string _Cor;
int _Pontos;

//Metodos
public:

NPC();
~NPC();
void setNome(string nome);
string getNome();
void setCor(string cor);
string getCor();
void setPontos(int pontos);
int getPontos();
string print();

};