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
- Průměrná přesnost: 78.31%
- Průměrná ztráta: 0.7391

## Vyhodnocení na testovacích datech

| Metrika | Výsledek |
|:--------|:---------|
| Accuracy | 69% |

- Nejmenší síť se ukázala jako nejstabilnější (malý rozptyl v boxplotu) a dosáhla nejlepšího výkonu.
- Zvětšování velikosti sítě nepřineslo zlepšení výsledků, což může být způsobeno omezenou velikostí trénovacích dat nebo přeučením větších modelů.

## Shrnutí
Experiment ukázal, že pro dataset GTSRB a jednoduché FFNN sítě:
- menší architektura je dostatečná,
- náhodnost inicializace vah má vliv – proto bylo vhodné provést 5 běhů,
- správné nastavení batch_size a počtu workers má významný vliv na rychlost trénování.
