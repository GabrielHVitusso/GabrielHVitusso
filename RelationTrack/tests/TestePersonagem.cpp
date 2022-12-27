#include "../include/Personagem.hpp"
#include <iostream>

using namespace std;

int main(){
    Personagem* teste1 = new Personagem();

    teste1->setNome("Douglas");

    teste1->print();

    teste1->setNome("Dismas");
    teste1->addNPC("testenpc", "vermelho");
    teste1->setPontos("testenpc", 5);

    teste1->print();

    teste1->setNome("Dismas_Com_1_NPC_A_Menos");
    teste1->removeNPC("testenpc");

    teste1->print();


    Personagem* teste2 = new Personagem("Gargulio");
    
    teste2->print();
    

    return 0;
}