# Experiment 04 – Trénink FFNN na GTSRB datasetu (PyTorch)

## Zadání
- Seznámení s tvorbou dopředné neuronové sítě (Feedforward Neural Network – FFNN) v PyTorch.
- Návrh, trénování a vyhodnocení 5 různých architektur na klasifikační úlize (GTSRB).
- Trénování každé architektury 5× pro vyrovnání vlivu náhodnosti.
- Vyhodnocení přesnosti a ztráty pomocí boxplotů.
- Vyhodnocení nejlepší topologie na testovacích datech.

## Parametry experimentu
- Dataset: GTSRB (German Traffic Sign Recognition Benchmark)
- Framework: PyTorch 2.7.0
- Batch size: 64
- Epochs: 2
- Počet běhů každé topologie: 5
- Learning rate: 0.0005
- Datum a čas: 14:34:31 Центральная Европа (лето) on Wednesday, May 28, 2025

## Použité architektury (topologie)
1. [128]
2. [256, 128]
3. [512, 256, 128]
4. [512, 256, 128, 64]
5. [1024, 512, 256, 128, 64]
Každá síť končí plně propojenou vrstvou na 43 výstupních tříd.

## Výsledky trénování

### Boxploty
Byly vytvořeny boxploty validační přesnosti a validační chyby (val_loss) přes 5 běhů pro každou architekturu.

**Boxplot validační přesnosti:**

![](../images\boxplot_accuracy.png)

**Boxplot validační chyby (loss):**

![](../images\boxplot_loss.png)

### Nejlepší model
- Nejlepší topologie na základě validační přesnosti: **Topology_2**
- Průměrná přesnost: 79.87%
- Průměrná ztráta: 0.7143

## Vyhodnocení na testovacích datech

| Metrika | Výsledek |
|:--------|:---------|
| Accuracy | 63% |

## Pozorování
- Nejmenší síť byla nejstabilnější (malý rozptyl v boxplotu) a dosáhla nejlepšího výkonu.
- Zvyšování velikosti sítě nevedlo k lepším výsledkům, což může být způsobeno omezenou velikostí trénovacích dat nebo přeučením větších modelů.

## Shrnutí
Experiment potvrdil, že pro GTSRB dataset a jednoduché FFNN sítě:
- menší architektura je dostatečná,
- náhodnost inicializace vah má vliv – proto bylo vhodné použít 5 běhů,
- správné nastavení batch_size a workers výrazně ovlivňuje rychlost trénování.
