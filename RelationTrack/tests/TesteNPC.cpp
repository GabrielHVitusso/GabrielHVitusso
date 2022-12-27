#include "../include/NPCs.hpp"
#include <iostream>

using namespace std;

int main(){
    NPC* teste = new NPC("teste");

    teste->print();

    teste->setCor("vermelho");
    teste->setPontos(10);
    teste->setNome("TESTE");

    cout << teste->getCor() << endl;
    cout << teste->getNome() << endl;
    cout << teste->getPontos() << endl; 

    teste->print();

    return 0;
}