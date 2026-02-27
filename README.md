# FixedToothDecayProject
new version


- sans_modified.parquet - dane po preczytaniu danych buchwalda; z wyciętymi próbkami na granicy zdrowe/chore_sztucznie
- 

Questions:
- czy my dobrze labelujemy zęby zdrowe/ chore sztucznie? bo może się okazać że buchwald też dziwnie obrócił tego zęba

- augmented są tylko chore naturalnie, czy jakoś zmieniamy tebelki bo obecnie 2 razy jest dokładnie ten sam model w zdrowe/chore sztucznie? ewentualnei augmentujemy też cgore sztucznie?

- wbrew temu co w paperze napisaliśmy to my robimy 80/20 train test split co moż edawać data leak

- normalisation w plotiwaniu budzi moje wątpliwości. to nie przywraca w pewien sposób tego co chcieliście żebym usunął że kolory na plotach nie mówią nic o predykcji między dwoma plotami? może normalizację między wszystkimi fieldami byłaby lepsza?

- obecnie dla plotów używaliśmy do treningu wszystkich danych przez to dłużo wolniej to się trenuje, jak będziemy mieli więcej danych to może będziemy mogli z tego zrezygnować

- czy dobrze rozumiem że w oryginalnym projekcie deepsety używają tylko polaryzacji v aboslutnie wszędzie? usunąłęm to.
