# 02 — Transcripción y recuperación

## La transcripción enchufable

El pipeline empieza desde voz, pero la conversión de voz a texto (ASR) es un
componente que cambia rápido y conviene poder intercambiar: un Whisper en la
nube, otro ASR, un modelo local, o un sistema previo que ya dio el texto. Además,
en pruebas y demos no hay audio ni red.

Por eso definimos una interfaz pequeña, `Transcriber`:

```python
def transcribe(self, audio_path: str) -> Transcript: ...
```

Con dos implementaciones:

- **`WhisperTranscriber`:** envía el fichero de audio a un endpoint compatible con
  OpenAI (`/audio/transcriptions`).
- **`ProvidedTranscript`:** trata su entrada como texto ya reconocido. Es lo que
  usan las pruebas, la demo y los pipelines alimentados por un ASR previo.

El pipeline depende **solo** de la interfaz. Cualquier fallo se convierte en
`TranscriptionError`. Así, todo el pipeline corre sin audio ni red (con
`ProvidedTranscript`) y cambiar de motor de ASR es implementar una interfaz, no
tocar el pipeline.

## La recuperación (retrieval) con TF-IDF

Para **fundamentar** la respuesta hay que recuperar los pasajes más relevantes a
la pregunta. Había dos caminos:

- **Embeddings densos:** un modelo que convierte texto en vectores + una base de
  datos vectorial. Potente, pero añade un modelo que alojar y más infraestructura.
- **Recuperación dispersa (TF-IDF / BM25):** clásica, por coincidencia de
  palabras.

Para esta base de conocimiento (unas pocas decenas de preguntas frecuentes,
cortas y distintivas), lo que importa es ser explicable, determinista y sin
dependencias externas. Así que elegimos **TF-IDF con similitud coseno**
(scikit-learn).

### Cómo funciona

1. Se vectoriza una vez la base (pregunta + respuesta de cada entrada) con TF-IDF.
2. La consulta se vectoriza y se compara con todas las entradas por coseno.
3. Se devuelven los `top_k` pasajes con mayor similitud, cada uno con su
   puntuación.

TF-IDF da más peso a las palabras distintivas y menos a las comunes; con
unigramas y bigramas, frases como "late payment" cuentan como señal.

### Ventajas

- **Explicable:** cada pasaje trae su puntuación de por qué se recuperó.
- **Sin infraestructura extra:** nada de modelo de embeddings ni base vectorial.
- **Determinista:** misma consulta, mismos pasajes; pruebas reproducibles.

### La contrapartida (honesta)

TF-IDF empareja **palabras**, no significado, así que puede fallar con paráfrasis
que no comparten vocabulario. La interfaz `retrieve` está pensada para poder
cambiar a un recuperador por embeddings más adelante sin tocar el resto, si el
corpus crece y lo necesita.

## Para fijar la idea

- Transcripción tras una interfaz `Transcriber`: enchufable, probable sin red.
- Recuperación TF-IDF: simple, explicable, determinista, sin infraestructura.
- Los pasajes recuperados son la base que luego permite evaluar la respuesta.
