# Guía de estudio — VoiceQA

Esta carpeta es una guía en español para entender VoiceQA a fondo. El código, los
comentarios y la documentación técnica están en inglés (nivel B2) a propósito,
para que el proyecto sea creíble y sirva para practicar. Esta guía explica el
"por qué" de cada decisión en tu idioma, para estudiar con calma.

## Orden sugerido de lectura

1. **[01-pipeline-y-grounding.md](01-pipeline-y-grounding.md)** — las cuatro
   etapas del pipeline y la idea central: fundamentar (grounding) y luego evaluar.
2. **[02-transcripcion-y-recuperacion.md](02-transcripcion-y-recuperacion.md)** —
   la transcripción enchufable y la recuperación TF-IDF.
3. **[03-respuesta-y-evaluacion.md](03-respuesta-y-evaluacion.md)** — cómo se
   genera la respuesta fundamentada y cómo se evalúa (grading) con LLM o
   heurística.
4. **[04-arquitectura-y-diseno.md](04-arquitectura-y-diseno.md)** — cómo encajan
   las piezas y qué decisiones de diseño se tomaron.
5. **[05-preguntas-de-entrevista.md](05-preguntas-de-entrevista.md)** — preguntas
   típicas y cómo responderlas con seguridad.
6. **[06-validacion-del-grader.md](06-validacion-del-grader.md)** — cómo se
   **mide** que el grader sirve: datos reales de habla transcrita (Spoken-SQuAD),
   la métrica ROC-AUC con su fórmula y un ejemplo trabajado, los dos controles
   negativos y cómo se deriva el veredicto.

## Resumen en una frase

VoiceQA responde preguntas habladas: transcribe el audio, recupera pasajes de una
base de conocimiento, genera una respuesta fundamentada solo en esos pasajes con
un LLM, y evalúa esa respuesta (qué tan fundamentada y relevante es) para
marcarla como fiable o derivarla a un humano — con alternativas sin conexión en
cada etapa.
