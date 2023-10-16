# Neural-Network

Lab02\
Sieci Neuronowe

Wynikiem implementacji listy jest program podzielony na 3 pliki:

- Functions.py -- funkcje pomocnicze, dyskretyzacja danych,
  normalizacja danych, obliczenie metryk dla zestawu testowego.

- Main.py -- środowisko testowe, w nim testuje wizualizuje
  zaimplementowane metody i funkcje.

- NeuralNetwork.py -- klasa prostej sieci neuronowej, opartej na
  gradiencie z entropii krzyżowej.

Implementacja rozwiązania:

Wyjście w sieci było implementowane na wzór:

𝑝(𝑥) = 𝜎(𝑊𝑥 + 𝑏)

    def p(self, x):
        argument = np.dot(x, self.W) + self.b\
        return self.sigmoid(argument)

gdzie funkcja sigmoid to:

<div style="text-align:center;">
  $$ \sigma(n) = \frac{1}{1 + e^{-n}} $$
</div>

Co w pythonie może być osiągnięte za pomocą funkcji
[expit(x)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html)
:

    def sigmoid(self, n):\
    return expit(n)

Jako funkcję kosztu wykorzystujemy entropię krzyżową:
<div style="text-align:center;">
    𝐿 = −𝑦 ln 𝑝(𝑥) − (1 − 𝑦) ln(1 − 𝑝(𝑥))
</div>

    def cross_entropy_loss(self, y, y_pred):
        epsilon = 1e-15
        loss = y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred +epsilon)
        return -np.sum(loss)

epsilon został dodany ponieważ był problem z liczeniem wartości 0.

Implementacja gradientu będzie za pomocą pochodnej po wagach modelu, co
z entropii krzyżowej daje:
<div style="text-align:center;">
  $$
  \frac{\partial L}{\partial w_i} = -(y - p(x))x
  $$
  $$
  \frac{\partial L}{\partial w_i} = (p(x) - y)x
  $$
</div>

    def compute_gradient(self, X_train, y_train):\
        y_pred = self.p(X_train)\
        dz = y_pred - y_train\
        dw = np.dot(X_train.T, dz)\
        db = np.sum(dz)\
        return dw, db

Model uczy się na podstawie zmiany wag, tak aby iść w stronę wyznaczoną
przez gradient. Implementacja:

𝑤~i~′ = 𝑤~𝑖~ − 𝛼 $\frac{\partial L\ }{\partial wi}$

    dw, db = self.compute_gradient(X_train, y_train)

Aktualizacja wag i bias zgodnie z gradientem i współczynnikiem
uczenia:

    self.W -= self.learning_rate * dw
    self.b -= self.learning_rate * db

Mój model posiada również 3 funkcje które mogą go wyuczyć:

- Fit_model_covergence -- który mówi o wystarczająco małej zmianie aby
  przerwać proces uczenia.

- Fit -- podstawowa wersja ucząca model, przechodząca przez cały zbiór
  X razy.

- Fit_batches -- wersja rozszerzona o dzielenie zbioru na paczki o
  nadanej wielkości, przechodzi przez zbiór X razy.

Aby zmaksymalizować uczenie się modelu, dane przed każdą iteracją są
losowo mieszane.

Aby zobaczyć wpływ procesowania danych będę rozpatrywał wszystkie wyniki
kontekście 3 metod procesowania:

- Normalizacja

- Dyskretyzacja

- Surowe dane

Hiperparametry i parametry zostały wybrane dla każdego osobno tak aby
zmaksymalizować ich potencjał.

Wyniki uczenia dla parametrow i hiperparametrów:

**Surowe dane**

    learning_rate_basic_without_b = 0.1
    learning_rate_basic_with_b = 0.001
    num_of_iterations_basic = 400
    batch_size = 100

![A graph with a line Description automatically
generated](media/basic_data_1.png){width="1.5384612860892388in"
height="1.1545067804024496in"}![A graph with colored lines and white
text Description automatically
generated](media/basic_data_2.png){width="1.5538462379702538in"
height="1.1727012248468942in"}![A graph with a line Description
automatically generated](media/basic_data_3.png){width="1.5769225721784776in"
height="1.1923884514435696in"}![A graph of a training metrics
Description automatically
generated](media/basic_data_4.png){width="1.5307688101487313in"
height="1.1597047244094487in"}

![A graph with blue lines Description automatically
generated](media/basic_data_5.png){width="1.561538713910761in"
height="1.1800634295713035in"}![A graph of different colored lines
Description automatically
generated](media/basic_data_6.png){width="1.561537620297463in"
height="1.1771216097987751in"}![A graph with a line Description
automatically generated](media/basic_data_7.png){width="1.6020122484689414in"
height="1.2in"}![A graph of a training metrics Description automatically
generated](media/basic_data_8.png){width="1.6in"
height="1.1968974190726158in"}

Możemy stąd zauważyć, że dane paczkowane, mają lepszy wynik ale są mniej
stabilne jeśli chodzi o metryki i proces uczenia.

Jendak oba wyniki są bardzo dobre, plasują się na poziomie \>=0.6 jeśli
chodzi o wszystkie metryki, co jest lepsze niż losowe zgadywanie.
Najlepsza

**Dane poddane dyskretyzacji:**

  learning_rate_discrete_without_b = 0.0005/
  learning_rate_discrete_with_b = 0.0005/
  batch_size = 64/
  num_of_iterations_discretization = 60

![A graph with a line Description automatically
generated](media/discretize_data_1.png){width="1.5384612860892388in"
height="1.1545067804024496in"}![A graph with colored lines and white
text Description automatically
generated](media/discretize_data_2.png){width="1.5538462379702538in"
height="1.1727012248468942in"}![A graph with a line Description
automatically generated](media/discretize_data_3.png){width="1.5769225721784776in"
height="1.1923884514435696in"}![A graph of a training metrics
Description automatically
generated](media/discretize_data_4.png){width="1.5307688101487313in"
height="1.1597047244094487in"}

![A graph with blue lines Description automatically
generated](media/discretize_data_5.png){width="1.561538713910761in"
height="1.1800634295713035in"}![A graph of different colored lines
Description automatically
generated](media/discretize_data_6.png){width="1.561537620297463in"
height="1.1771216097987751in"}![A graph with a line Description
automatically generated](media/discretize_data_7.png){width="1.6020122484689414in"
height="1.2in"}![A graph of a training metrics Description automatically
generated](media/discretize_data_8.png){width="1.6in"
height="1.1968974190726158in"}


**Dane poddane normalizacji:**
  learning_rate_normalization_without_b = 0.001\
  learning_rate_normalization_with_b = 0.001\
  num_of_iterations_normalization = 200\
  batch_size= 128

![A graph with a line Description automatically
generated](media/normalize_data_1.png){width="1.813377077865267in"
height="1.3384612860892389in"}![A graph of different colored lines
Description automatically
generated](media/normalize_data_2.png){width="1.769628171478565in"
height="1.3230774278215223in"}![A graph of a training cost Description
automatically generated](media/normalize_data_3.png){width="1.7570931758530184in"
height="1.3153849518810148in"}![A graph of a training metrics
Description automatically
generated](media/normalize_data_4.png){width="1.7414840332458443in"
height="1.3076924759405075in"}![A graph with a line Description
automatically generated](media/normalize_data_5.png){width="1.7569444444444444in"
height="1.3218219597550307in"}![A graph of a training metrics
Description automatically generated with medium
confidence](media/normalize_data_6.png){width="1.8051049868766404in"
height="1.3461537620297463in"}![A graph with a line Description
automatically generated](media/normalize_data_7.png){width="1.7665955818022747in"
height="1.3307699037620297in"}![A graph of a training metrics
Description automatically
generated](media/normalize_data_8.png){width="1.7742454068241469in"
height="1.3307688101487314in"}

W tym przypadku możemy zauważyć, że jest zdecydowanie mniej iteracji bo
tylko 60 i model się stabilizuje, w przypadku paczkowania pomimo braku
wyraźnej różnicy na wykresie kosztu możemy zauważyć różnicę w miarach.
Wszystkie testowe miary wskazują ok. 90% poprawności, co jest wspaniałym
wynikiem.

W przypadku paczkowania możemy także zauważyć mniejsze wahania metryk.

Wnioski:

Dobranie odpowiednich parametrów i hiperparametrów odgrywa kluczową rolę
w powodzeniu modelu. Jest to ciężkie bez znajomości metod na znalezienie
optmalnych współczynników, trzeba sprawdzać to metodą prób i błędów.

Preprocessing też odgrywa ważną rolę w tym jak sprawuje się model.

Nawet taki prosty model może sobie dobrze radzić z binarną klasyfikacją.

