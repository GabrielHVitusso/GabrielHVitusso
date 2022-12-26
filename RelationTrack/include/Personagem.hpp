#include <iostream>
#include <map>
#include "NPCs.hpp"

using namespace std;

class Personagem{
//Variaveis
string Nome;
public:
map <string, NPC> NPCs;

//Metodos

Personagem();
~Personagem();
string print();
};