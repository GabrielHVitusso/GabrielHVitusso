#include <iostream>
#include "../third_party/colors.hpp"
#include "../include/NPCs.hpp"

using namespace std;

NPC::NPC(string nome){
    setNome(nome);
    setCor("branco");
    _Pontos = 0;
}

NPC::NPC(){}

NPC::~NPC(){}

void NPC::setNome(string nome){
    _Nome = nome;
}

string NPC::getNome(){
    return _Nome;
}

void NPC::setCor(string cor){
    _Cor = cor;
}

string NPC::getCor(){
    return _Cor;
}

void NPC::setPontos(int pontos){
    _Pontos = pontos;
}

int NPC::getPontos(){
    return _Pontos;
}

void NPC::print(){

    cout <<"    ";

    if(_Cor == "branco"){
        cout << colors::white << _Nome;
    }else if(_Cor == "verde"){
        cout << colors::green << _Nome;
    }else if(_Cor == "vermelho"){
        cout << colors::red << _Nome;
    }else if(_Cor == "amarelo"){
        cout << colors::yellow << _Nome;
    }else if(_Cor == "azul"){
        cout << colors::blue << _Nome;
    }else if(_Cor == "roxo"){
        cout << colors::magenta << _Nome;
    }else if(_Cor == "ciano"){
        cout << colors::cyan << _Nome;
    }else{
        cout << colors::on_red  << colors::grey <<  "ERRO, COR INVALIDA\n" << _Nome;
    }


    cout << colors::reset << " " << _Pontos << endl;
}