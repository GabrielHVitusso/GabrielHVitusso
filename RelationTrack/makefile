CC := g++
SRCDIR := src
BUILDDIR := build
TARGETSLAVE := slave.exe
TARGETMASTER := master.exe
CFLAGS := -g -Wall -std=c++11 -I include/

all: main

NPC:

	$(CC) $(CFLAGS) -c src/NPCs.cpp -o build/NPCs.o

Personagem:

	$(CC) $(CFLAGS) -c src/Personagem.cpp -o build/Personagem.o

main: NPC Personagem

	$(CC) $(CFLAGS) build/NPCs.o tests/TesteNPC.cpp -o bin/TesteNPC.exe
	$(CC) $(CFLAGS) build/NPCs.o build/Personagem.o tests/TestePersonagem.cpp -o bin/TestePersonagem.exe
