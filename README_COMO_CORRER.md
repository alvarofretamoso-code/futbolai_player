# FutAI — Player Analyzer

## Setup (una sola vez)

```bash
cd fut-ai-cladude
pip install -r requirements.txt
```

## Correr la app

```bash
uvicorn app:app --reload --port 8000
```

Abrí el browser en: **http://localhost:8000**

## Uso

1. Ingresá tu **Gemini API Key** (obtenela en https://aistudio.google.com/app/apikey)
2. Subí el video del partido
3. En el primer frame, **dibujá un rectángulo** sobre el jugador que querés analizar
4. Hacé click en **"Analizar jugador"** y esperá
5. Cuando termina, podés **exportar el Excel** con el resumen y timeline

## Cómo funciona

- **Tracking:** OpenCV CSRT — sigue al jugador frame a frame (2 fps de muestreo)
- **Análisis:** Gemini 1.5 Flash recibe batches de 8 frames CON el bounding box dibujado en rojo
- **Anti-alucinación:** umbrales de confianza ≥75%, deduplicación de acciones (mismo evento no cuenta dos veces en < 1.5s)

## Tips para mejores resultados

- Videos donde el jugador sea **bien visible** (no cortado por el borde)
- Encuadre del rectángulo **ajustado al cuerpo** del jugador (no demasiado grande ni pequeño)
- Videos con **buena resolución** y sin mucho motion blur
- Si el tracker pierde al jugador (se mueve muy rápido o queda tapado), los frames de esa sección simplemente se omiten

## Estructura

```
fut-ai-cladude/
├── app.py              # Backend FastAPI
├── requirements.txt
├── uploads/            # Videos subidos + Excel generados (creado automáticamente)
└── static/
    └── index.html      # Frontend
```
