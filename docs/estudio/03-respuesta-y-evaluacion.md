# 03 — La respuesta y la evaluación

## Generar la respuesta (answerer)

El `Answerer` recibe la pregunta y los pasajes recuperados. Construye un prompt
que le dice al LLM:

> "Responde usando ÚNICAMENTE el contexto. Si la respuesta no está en el
> contexto, di exactamente 'I don't have that information'. Una o dos frases."

Así la respuesta queda **fundamentada** en los pasajes. Devuelve un `Answer` con
el texto, las **citas** (los temas de los pasajes usados) y la `source`.

### El respaldo (fallback)

Si el LLM no está disponible (cascada agotada), el answerer devuelve el texto del
**mejor pasaje recuperado** tal cual. Es correcto (viene de la base de
conocimiento), aunque menos fluido. `source = "retrieval"`. Si no hay pasajes,
devuelve "I don't have that information".

## Evaluar la respuesta (grader)

El `Grader` es la **puerta de calidad** del pipeline. Decide dos cosas:

- **Groundedness (fundamentación):** ¿la respuesta está respaldada por el
  contexto? (0.0–1.0)
- **Relevancia:** ¿la respuesta aborda la pregunta? (sí/no)

Y combina: `confident = groundedness >= umbral AND relevant`.

### El camino LLM: JSON estricto

Al LLM se le pide **solo** un JSON:

```json
{"groundedness": <0.0-1.0>, "relevant": <true|false>, "rationale": "<una frase>"}
```

Se extrae el primer bloque `{...}`, se parsea, y se **acota** la groundedness al
rango [0,1] (si el modelo devuelve 5, se vuelve 1.0). Validar, no confiar.

### El respaldo heurístico

Si el LLM falla o devuelve algo ilegible, se estima sin red con señales de
solapamiento de tokens (ignorando palabras vacías y muy cortas):

- **groundedness** = fracción de palabras de contenido de la *respuesta* que
  aparecen en el *contexto* recuperado. Mide "¿está respaldada por la fuente?".
- **relevancia** = se decide con **dos señales complementarias**, y basta con que
  **una** se cumpla con claridad (si ambas son débiles, se deriva):
  1. **cobertura léxica:** qué fracción de las palabras de la *pregunta* cubre el
     *contexto* recuperado (umbral `_MIN_RELEVANCE_OVERLAP = 0.15`). Una sola
     palabra compartida no basta —eso marcaba como relevante una respuesta de
     ciclo de facturación ante una pregunta de cambio de plan—; un solapamiento
     real sí.
  2. **confianza de recuperación:** la mejor puntuación de coseno de los pasajes
     (umbral `_MIN_RETRIEVAL_SCORE = 0.30`). Acepta una recuperación segura aunque
     las palabras exactas difieran, y una mejor puntuación baja significa que la
     recuperación no encontró nada que de verdad encaje.

El sesgo es **deliberado**: una escalada falsa (un humano revisa una respuesta que
era buena) es barata; un "confident" falso (una respuesta equivocada servida como
correcta) es caro. Por eso ante la duda se escala.

Nota importante: la relevancia compara la **pregunta** con el **contexto
recuperado**, no la respuesta con la pregunta. Así se pregunta lo correcto: "¿la
recuperación encontró de verdad material que cubra esta pregunta?".

## Un ejemplo que enseña mucho: la pregunta fuera de tema

Pregunta: "¿Cuál es la capital de Francia?" (no está en la base telecom).

- El recuperador devuelve el pasaje menos malo (p. ej. "billing_cycle"), pero con
  una **puntuación baja** (~0,17, por debajo de 0,30).
- El answerer (en fallback) devuelve ese pasaje. Su groundedness es alta (el texto
  **es** un pasaje, así que todas sus palabras están en el contexto).
- Pero las **dos señales de relevancia** fallan: la cobertura de "capital de
  Francia" en el contexto telecom es casi nula (< 0,15) **y** la mejor puntuación
  de recuperación es baja (< 0,30).
- Resultado: `relevant = False` → `confident = False` → **derivar a un humano**.

Es decir: la **relevancia** es la que atrapa lo fuera de tema, aunque la
fundamentación sea alta. Y como se apoya en dos señales, también atrapa el caso
sutil de una respuesta fundamentada que comparte una palabra suelta con la
pregunta pero no la aborda de verdad.

## Transparencia

Cada resultado guarda `answer_source` y `grade_source` ("llm" o
"heuristic"/"retrieval"). Quien lee sabe cómo se obtuvo cada cosa. La degradación
es **visible**, no oculta.

## Para fijar la idea

- Answerer: responde solo con el contexto; si no hay LLM, devuelve el mejor
  pasaje.
- Grader: puntúa fundamentación y relevancia; con LLM (JSON validado) o
  heurística.
- Relevancia heurística = cobertura léxica **o** confianza de recuperación; basta
  con una, y ante la duda se escala.
- `confident = groundedness ≥ umbral AND relevant`.
- La relevancia atrapa lo fuera de tema aunque la fundamentación sea alta.
