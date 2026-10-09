# 06 — Validar el grader con datos reales (ROC-AUC y controles)

Esta guía explica cómo se **mide** si el grader de VoiceQA sirve de verdad. No
basta con que el grader diga "confío en esta respuesta": hay que demostrar, con
datos reales y etiquetas de verdad, que su puntuación **distingue** las respuestas
correctas de las incorrectas. Aquí está el método, la fórmula y los números
reales.

Código: [`voiceqa/real_data.py`](../../voiceqa/real_data.py) (carga y etiquetado)
y [`voiceqa/real_benchmark.py`](../../voiceqa/real_benchmark.py) (medición).
Ejecuta `python -m voiceqa.real_benchmark --config default`.

## 1. ¿Qué tarea evalúa el grader?

El grader recibe `(pregunta, respuesta, pasajes)` y devuelve una **groundedness**
(fundamentación) entre 0 y 1: qué fracción de las palabras de contenido de la
*respuesta* aparecen en el *contexto* recuperado. Es la señal continua que
decide, junto con la relevancia, si la respuesta se marca `confident` o se deriva
a un humano.

La pregunta que queremos responder es concreta y medible:

> **¿La puntuación de groundedness es alta cuando la respuesta es correcta y baja
> cuando es incorrecta?**

Si lo es, el grader mide algo real. Si no, es ruido disfrazado de confianza.

## 2. Los datos: Spoken-SQuAD (habla transcrita de verdad)

Usamos [`alinet/spoken_squad`](https://huggingface.co/datasets/alinet/spoken_squad):
preguntas de SQuAD 1.1 sobre pasajes que se **convirtieron a voz y se
transcribieron de vuelta con un sistema ASR**. Por eso el campo `context` es un
pasaje hablado con errores reales de reconocimiento (p. ej. *"in 1858"* aparece
como *"eighteen fifty eight"*). Es exactamente la entrada para la que VoiceQA
está pensado: una respuesta evaluada contra un pasaje recuperado del habla.

Hay dos niveles de ruido, a propósito:

- `default` (train): ASR de menor error (~22% WER).
- `WER44` (test): ASR de estrés (~44% WER) — más palabras mal transcritas.

## 3. ¿Cómo se deriva la etiqueta correcto/incorrecto?

De cada registro `(pregunta, respuesta_oro, pasaje_transcrito)` construimos **dos**
ejemplos para el grader (ver `build_examples`):

- **Correcto (`label = 1`):** la respuesta es la **respuesta de oro** de ese
  registro. Pertenece a ese pasaje, así que una comprobación de fundamentación
  que funcione debería puntuarla alto.
- **Incorrecto (`label = 0`):** la misma pregunta y el mismo pasaje, pero con una
  **respuesta ajena** — la respuesta de oro de otro registro, el `(i + 37) mod n`.
  Es una respuesta real, pero no pertenece a este pasaje, así que debería puntuar
  bajo.

Si por casualidad la respuesta ajena coincide con la de oro (respuestas repetidas
entre preguntas), ese registro **no** aporta ejemplo negativo: así nunca entra un
par mal etiquetado. La etiqueta es, por tanto, **la verdad de referencia**; la
groundedness del grader es lo que se pone a prueba contra ella.

## 4. La métrica principal: ROC-AUC (con fórmula)

El problema de medir con *accuracy* a secas es que depende de un umbral. ROC-AUC
es **independiente del umbral** y tiene una interpretación probabilística
preciosa:

> **AUC = P( score(correcto) > score(incorrecto) )**

Es decir: toma al azar un ejemplo correcto y uno incorrecto; la AUC es la
probabilidad de que el grader le dé **más** puntuación al correcto. Los empates
cuentan como medio:

$$
\mathrm{AUC} = \frac{1}{|P|\,|N|} \sum_{i \in P} \sum_{j \in N}
\Big[ \mathbf{1}(s_i > s_j) + \tfrac{1}{2}\,\mathbf{1}(s_i = s_j) \Big]
$$

donde `P` son los correctos, `N` los incorrectos y `s` la puntuación de
groundedness.

- **AUC = 0.5** → azar: el grader no distingue nada.
- **AUC = 1.0** → separación perfecta: todo correcto puntúa por encima de todo
  incorrecto.
- **AUC = 0.0** → separación perfecta pero **invertida** (el grader tiene el signo
  al revés).

El código lo calcula con rangos promedio (identidad de Mann-Whitney U), que es lo
mismo pero en `O(n log n)` y exacto con empates. El test
`test_roc_auc_matches_sklearn_on_random_data` comprueba que coincide con
scikit-learn.

### Un ejemplo trabajado con la forma de los números reales

En los datos reales los correctos se amontonan cerca de 1.0 (la respuesta suele
estar literalmente en el pasaje) y los incorrectos cerca de 0.0. Tomemos un mini
conjunto con esa forma, incluyendo un empate en 0.5:

- correctos `P = [1.0, 1.0, 0.5]`
- incorrectos `N = [0.0, 0.0, 0.5]`

Hay `|P| × |N| = 3 × 3 = 9` pares. Contamos para cada correcto cuántos
incorrectos "gana" (empate = ½):

| correcto | vs 0.0 | vs 0.0 | vs 0.5 | suma |
| --- | --- | --- | --- | --- |
| 1.0 | 1 | 1 | 1 | 3.0 |
| 1.0 | 1 | 1 | 1 | 3.0 |
| 0.5 | 1 | 1 | ½ | 2.5 |

Total = `3.0 + 3.0 + 2.5 = 8.5`. Entonces:

$$
\mathrm{AUC} = \frac{8.5}{9} = 0.944
$$

El único "descuento" respecto a 1.0 viene del empate en 0.5 (ese caso límite que,
en el pipeline real, es donde el grador léxico duda y conviene un grader LLM).

### Los números reales

| Métrica (~760 ejemplos) | `default` (~22% WER) | `wer44` (~44% WER) |
| --- | --- | --- |
| **ROC-AUC** (held-out) | **0.987** (0.994) | **0.849** (0.841) |
| media correcto vs incorrecto | 0.986 vs 0.062 | 0.712 vs 0.098 |
| accuracy / precision / recall de la puerta | 0.934 / 0.925 / 0.945 | 0.798 / 0.875 / 0.697 |
| ρ de Spearman (score vs token-F1) | 0.946 | 0.672 |

La lectura importante: con transcripción limpia el grader léxico separa casi
perfecto (0.987); cuando sube el error del ASR cae a 0.849 **de forma
predecible**. ¿Por qué cae el *recall*? Porque cuando el ASR destroza las propias
palabras de la respuesta (una fecha dicha en voz alta, un nombre raro), la
respuesta correcta ya **no** está presente léxicamente en el pasaje, el grader no
puede confirmarla y, con buen criterio, la deriva. Ese es el límite exacto del
grader léxico: suficiente para audio limpio y triaje de alto recall; para audio
muy degradado o respuestas correctas pero reformuladas, conviene activar el camino
del grader LLM (ya está en `grader.py`) y volver a correr este mismo benchmark
para cuantificar la mejora. La medición **es** la capacidad: te dice dónde deja de
bastar el grader barato.

## 5. Los dos controles negativos (y por qué deben colapsar)

Un número alto no demuestra nada por sí solo: podría ser un artefacto. Por eso lo
sometemos a dos **controles negativos**. Son experimentos diseñados para
**destruir** la señal; si la AUC sigue alta tras destruirla, es que medíamos algo
espurio.

1. **Barajar las etiquetas (shuffled labels).** Mantenemos las puntuaciones
   reales del grader, pero permutamos al azar qué ejemplo es correcto y cuál
   incorrecto. Si la puntuación no tiene nada que ver con la verdad una vez
   barajada, la AUC debe caer a ~0.5. *Debe colapsar* porque rompemos la relación
   entre puntuación y etiqueta.

2. **Referencia aleatoria (random reference passage).** Volvemos a evaluar cada
   respuesta, pero contra un pasaje **al azar de otro registro** (una permutación
   sin puntos fijos). La groundedness pierde su sentido: una respuesta rara vez
   está "fundamentada" en un pasaje que no es el suyo. *Debe colapsar* a ~0.5
   porque quitamos la señal que hacía que la puntuación fuera informativa.

Resultados reales (media de varias repeticiones):

| Control | `default` | `wer44` |
| --- | --- | --- |
| barajar etiquetas | 0.505 | 0.502 |
| referencia aleatoria | 0.505 | 0.497 |

Los dos caen a ~0.50. Eso es justo lo que queremos ver: la puntuación del grader
solo conserva su poder de discriminación cuando **tanto las etiquetas como los
pasajes son los verdaderos**. Una señal que sobrevive a ambos controles mide algo
real.

## 6. ¿Cómo se deriva el veredicto?

El benchmark declara el resultado **VALIDATED** solo si se cumplen las tres
condiciones a la vez (ver `Report.validated`):

1. **Supera el azar con claridad:** AUC real ≥ 0.70.
2. **Los dos controles colapsan:** cada control a ≤ 0.1 de 0.5 (es decir, entre
   0.40 y 0.60).
3. **Hay un hueco claro:** AUC real − el mayor de los controles ≥ 0.20.

Con los números reales: `default` da AUC 0.987 con controles 0.505/0.505 (hueco
~0.48), y `wer44` da AUC 0.849 con controles 0.502/0.497 (hueco ~0.35). Ambos
cumplen las tres condiciones → **VALIDATED**. No es un 1.0 inflado: es una AUC
modesta-pero-real en el caso difícil y muy alta en el fácil, y en los dos casos
**bate a sus propios controles** con holgura. Eso es exactamente lo que valida un
grader: no un número grande, sino un número que mide la verdad y se derrumba
cuando le quitas la verdad.

## Para fijar la idea

- Validar un grader = medir si su puntuación **distingue** correcto de incorrecto
  con datos reales y etiquetas de verdad.
- Métrica principal: **ROC-AUC = P(score(correcto) > score(incorrecto))**, sin
  umbral; 0.5 es azar, 1.0 es perfecto.
- Etiqueta: correcto = respuesta de oro del propio pasaje; incorrecto = respuesta
  de oro ajena en el mismo pasaje.
- Dos controles (barajar etiquetas, referencia aleatoria) **deben** colapsar a
  ~0.5; si no, la señal era espuria.
- Veredicto: supera el azar **y** los controles colapsan **y** hay hueco claro.
- El grader léxico es fuerte con ASR limpio (0.987) y se degrada con ASR ruidoso
  (0.849): la medición señala cuándo conviene añadir un grader LLM.
