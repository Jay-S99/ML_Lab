# Odpowiedzi teoretyczne i zakres wykonany bez nagrań

## Rozpoznawanie komend a transkrypcja dowolnych zdań

Klasyfikator komend wybiera jedną etykietę z małego, zamkniętego zbioru: `start`, `stop`, `lewo`, `prawo` lub `skok`. Nie rozpoznaje kolejności fonemów ani nie tworzy tekstu. Jego warstwa wyjściowa ma dokładnie pięć wyników, dlatego nawet ciszę lub nieznane słowo spróbuje przypisać do jednej z tych klas.

System transkrypcji mowy rozpoznaje sekwencję elementów języka i buduje zdanie o zmiennej długości. Potrzebuje znacznie większego zbioru, modelu językowego, obsługi wielu możliwych słów oraz mechanizmu dekodowania kolejnych fragmentów wypowiedzi. Wynikiem jest tekst, a nie jedna etykieta.

## Dlaczego sama amplituda nie wystarcza

Wykres amplitudy pokazuje głośność i zmiany energii w czasie, ale słabo opisuje skład częstotliwościowy dźwięku. Dwa różne słowa mogą mieć podobną obwiednię amplitudy, a to samo słowo wypowiedziane ciszej lub głośniej może mieć bardzo różne wartości amplitudy. Na przebieg wpływają również mikrofon, odległość, szum i akustyka pomieszczenia.

MFCC opisują krótkookresowe widmo w skali zbliżonej do percepcji człowieka. Dzięki temu zachowują informacje związane z barwą i układem głosek, a są mniej zależne od samej głośności. Nadal nie są odporne na wszystkie zakłócenia, dlatego potrzebne są różnorodne dane i poprawna walidacja.

## Dlaczego podział odbywa się według mówców

Losowe dzielenie pojedynczych plików mogłoby umieścić niemal identyczne próbki tej samej osoby w train i test. Model częściowo rozpoznawałby wtedy głos lub warunki nagrania zamiast komendy. Rozłączne zbiory mówców lepiej sprawdzają, czy model uogólnia działanie na nowe osoby.

## Dlaczego augmentacja jest wykonywana dopiero po podziale

Oryginał i jego zaszumiona albo przesunięta kopia są mocno powiązane. Gdyby trafiły do różnych zbiorów, wynik walidacyjny lub testowy byłby zawyżony. Najpierw dzielone są ścieżki według mówców, a dopiero później augmentowane są wyłącznie elementy zbioru treningowego.

## Ograniczenia wyniku softmax

Softmax rozdziela 100% wyniku pomiędzy znane klasy. Wysoka wartość oznacza, że dana klasa wygrała z pozostałymi dla konkretnego modelu, ale nie jest gwarancją poprawności. Model może być bardzo pewny także dla ciszy, hałasu albo słowa spoza zbioru. Próg odrzucenia należy dobrać na walidacji; solidniejsze rozwiązanie wymaga klas `cisza` i `inne` oraz odpowiednich przykładów treningowych.

## Co wykonano bez nagrywania głosu

- Narzędzie wizualizacji amplitudy i MFCC, w tym porównanie 13, 20 i 40 współczynników.
- Raport liczebności, czasu trwania, częstotliwości próbkowania, ciszy i uszkodzonych WAV-ów.
- Augmentacje przez szum i przesunięcie w czasie, stosowane tylko do train.
- Architektury `base` i `extended`, wspólne dla treningu i predykcji, wraz z liczeniem parametrów.
- Podział train/validation/test według mówców, średnia strata epoki, wybór najlepszej epoki na walidacji i jednorazowy test.
- Zapisywanie historii, macierzy pomyłek, metryk, błędnych predykcji, czasu treningu i tabeli eksperymentów.
- Konfiguracja learning rate, batch size, MFCC, architektury, augmentacji i ziarna w menu oraz CLI.
- Test integracyjny na sztucznych tonach, który sprawdza kod, ale nie stanowi eksperymentu rozpoznawania mowy.

Pomiar niezależny od danych daje 4 965 trenowalnych parametrów dla wariantu `base` oraz 23 621 dla `extended`. Wariant rozszerzony ma więc około 4,76 razy więcej parametrów. Rzetelne porównanie czasu całego treningu należy wykonać na tym samym rzeczywistym zbiorze i podziale mówców, ponieważ czas obejmuje także liczbę przykładów, augmentację i wybraną liczbę MFCC.

## Czego nie można uczciwie zakończyć bez mowy

Nie można wyciągać wniosków o jakości rozpoznawania polskich komend, porównywać głosów dwóch osób, analizować pięciu rzeczywistych błędów ani wiarygodnie porównywać accuracy konfiguracji. Sztuczne tony służą wyłącznie do technicznego sprawdzenia przepływu danych.
