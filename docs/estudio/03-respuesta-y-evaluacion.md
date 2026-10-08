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

Si el LLM falla o devuelve algo ilegible, se estima con **solapamiento de
tokens**:

- groundedness = fracción de palabras de contenido de la respuesta que aparecen
  en el contexto;
- relevante = ¿hay solapamiento de palabras de contenido entre la respuesta y la
  pregunta?

(Se ignoran palabras vacías y muy cortas.)

## Un ejemplo que enseña mucho: la pregunta fuera de tema

Pregunta: "¿Cuál es la capital de Francia?" (no está en la base telecom).

- El recuperador devuelve el pasaje menos malo (p. ej. "billing_cycle").
- El answerer (en fallback) devuelve ese pasaje. Su groundedness es alta (el texto
  **es** un pasaje, así que todas sus palabras están en el contexto).
- Pero la **relevancia** es falsa: la respuesta no comparte palabras de contenido
  con "capital de Francia".
- Resultado: `confident = False` → **derivar a un humano**.

Es decir: la **relevancia** es la que atrapa lo fuera de tema, aunque la
fundamentación sea alta. Por eso hacen falta las dos señales.

## Transparencia

Cada resultado guarda `answer_source` y `grade_source` ("llm" o
"heuristic"/"retrieval"). Quien lee sabe cómo se obtuvo cada cosa. La degradación
es **visible**, no oculta.

## Para fijar la idea

- Answerer: responde solo con el contexto; si no hay LLM, devuelve el mejor
  pasaje.
- Grader: puntúa fundamentación y relevancia; con LLM (JSON validado) o
  heurística.
- `confident = groundedness ≥ umbral AND relevant`.
- La relevancia atrapa lo fuera de tema aunque la fundamentación sea alta.
