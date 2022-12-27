#include <iostream>
#include <map>
#include "NPCs.hpp"

using namespace std;

class Personagem{
//Variaveis
string _Nome;
public:
map <string, NPC> NPCs;

//Metodos

Personagem();
Personagem(string nome);
~Personagem();
void setNome(string nome);
void print();
void addNPC(string nome, string cor);
void setCor(string nome, string cor);
void setPontos(string nome, int pontos);
void removeNPC(string nome);
};