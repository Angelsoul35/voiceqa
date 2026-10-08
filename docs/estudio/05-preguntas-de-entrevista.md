# 05 — Preguntas de entrevista

Preguntas típicas sobre este proyecto y cómo responderlas con seguridad.
Practica diciéndolas en voz alta en inglés también.

## Sobre la idea central

**¿Qué problema resuelve VoiceQA que no resuelve "un modelo más grande"?**
El de responder con seguridad cuando no se debería. La solución no es un modelo
mayor, sino un pipeline que **fundamenta** la respuesta en una fuente conocida y
la **evalúa** antes de servirla. Una respuesta sin respaldo o fuera de tema se
marca para que la vea un humano.

**¿Por qué separas el que responde del que evalúa?**
Para que una respuesta que "suena segura" tenga que pasar una comprobación
independiente. Si el mismo componente respondiera y se autojuzgara, el sesgo de
confianza se colaría. Separarlos hace la evaluación creíble.

**¿Qué es "groundedness" y cómo la mides?**
Es cuánto se apoya la respuesta en el contexto recuperado. Con LLM, le pido un
JSON con un valor 0–1. Sin LLM, estimo con el solapamiento de palabras de
contenido entre la respuesta y el contexto.

## Sobre robustez

**¿Qué pasa si no hay LLM ni clave de API?**
El pipeline sigue: el answerer devuelve el mejor pasaje recuperado y el grader usa
una heurística de solapamiento. Cada etapa tiene respaldo, así que corre sin red
(así corren las pruebas y la demo). Y marca con qué camino se obtuvo cada cosa.

**¿Y la transcripción sin audio ni red?**
Uso `ProvidedTranscript`, que trata la entrada como texto ya reconocido. En
producción, `WhisperTranscriber` llama a un ASR real. Las dos implementan la misma
interfaz `Transcriber`.

## Sobre la recuperación

**¿Por qué TF-IDF y no embeddings?**
Para una base pequeña y distintiva, TF-IDF es explicable, determinista y no
necesita modelo ni base vectorial. La interfaz `retrieve` permite cambiar a
embeddings densos si el corpus crece. Elijo la herramienta adecuada al tamaño del
problema.

**¿Cómo evitas responder fuera de tema?**
La relevancia lo atrapa: aunque la fundamentación sea alta (porque la respuesta es
literalmente un pasaje), si la respuesta no comparte palabras de contenido con la
pregunta, `relevant = False` y se deriva. Por eso uso **dos** señales, no una.

## Sobre el diseño

**¿Por qué este proyecto no usa blockchain?**
Porque no lo necesita. Es orquestación de LLM con recuperación y evaluación.
Meter una cadena sería artificial. Buena ingeniería es elegir la herramienta
adecuada.

**¿Qué cubren las pruebas?**
Transcripción (offline y el guardia sin clave), recuperación, answerer (camino
LLM y fallback), grader (heurística, JSON del LLM y acotado), el pipeline
completo (en tema confiado vs. fuera de tema derivado), la API y la CLI. Ninguna
prueba toca la red. Cobertura mínima en CI del 80 %.

## Pregunta trampa

**Si subes el umbral de groundedness, ¿qué pasa?**
Más respuestas se marcan como baja confianza y se derivan: subes la precisión
(menos respuestas malas servidas) a costa de la cobertura (más derivaciones). Es
una palanca de producto: en un dominio sensible (salud, legal) la subiría; en uno
tolerante, la bajaría. Lo importante es que el umbral es explícito y configurable,
no un número escondido en el código.
