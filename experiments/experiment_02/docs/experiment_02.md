---
title: "Cvičení 4"
author: "Vaše jméno"
date: "2025-03-25"
---

# Záznam do deníku – Trénování neuronové sítě na funkci `sin(x)`

## Popis úlohy
Cílem experimentu bylo natrénovat jednoduchou plně propojenou neuronovou síť pro aproximaci matematické funkce `sin(x)`. Síť měla jednu skrytou vrstvu s proměnlivým počtem neuronů a lineární výstupní vrstvu.

## Parametry trénování
- **Vstupní vrstva**: 1 neuron
- **Výstupní vrstva**: 1 neuron
- **Aktivační funkce**: `tanh` (skrytá vrstva)
- **Počet epoch**: `1000`
- **Learning rate**: `0.01`
- **Ztrátová funkce**: `Mean Squared Error (MSE)`
- **Trénovací data**: `200 bodů` rovnoměrně rozložených na intervalu `[-2π, 2π]`

## Experiment: Počet neuronů ve skryté vrstvě
Provedl jsem experiment s následujícími velikostmi skryté vrstvy:
`[1, 2, 4, 8, 16]`
Pro každý model jsem zaznamenal průběh trénovací chyby a finální MSE.

## Výsledky
### Trénovací chyba (MSE) – posledních 100 epoch
![Boxplot trénovací chyby](../images/boxplot.png)

### Nejlepší model
- **Počet neuronů ve skryté vrstvě**: 16
- **Finální trénovací chyba (MSE)**: 0.1453

### Porovnání výstupu nejlepšího modelu s funkcí sin(x)
![Predikce nejlepšího modelu](../images/best_model_prediction.png)
