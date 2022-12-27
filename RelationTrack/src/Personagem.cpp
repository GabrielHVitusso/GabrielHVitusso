#include <iostream>
#include <map>
#include "../third_party/colors.hpp"
#include "../include/Personagem.hpp"

using namespace std;

Personagem::Personagem(){
    _Nome = "NULL";
}

Personagem::Personagem(string nome){
    setNome(nome);
}

Personagem::~Personagem(){
    _Nome = "";
    NPCs.clear();
}

void Personagem::setNome(string nome){
    _Nome = nome;
}

void Personagem::print(){
    cout << colors::grey << colors::on_white << _Nome << colors::reset << endl;

    for(auto i = NPCs.begin(); i!=NPCs.end(); i++){

        i->second.print();

    }

}

void Personagem::addNPC(string nome, string cor){
    NPC npc(nome);
    npc.setCor(cor);

    NPCs.insert(pair<string, NPC>(nome, npc));
}

void Personagem::setCor(string nome, string cor){
    NPCs[nome].setCor(cor);
}

void Personagem::setPontos(string nome, int pontos){
    NPCs[nome].setPontos(pontos);
}

void Personagem::removeNPC(string nome){
    NPCs.erase(nome);
}