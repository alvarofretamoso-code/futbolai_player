# FutAI — Football Player Analyzer

Analizá acciones de un jugador (pases, barridas, cabezazos, disparos) a partir de un video, usando IA.

## Requisitos

- Python 3.10 o superior
- Una API key de Gemini (gratis en https://aistudio.google.com/app/apikey)

## Instalación (una sola vez)

```bash
# 1. Clonar el repo
git clone https://github.com/alvarofretamoso-code/futbolai_player.git
cd futbolai_player

# 2. Crear entorno virtual
python -m venv venv

# 3. Activarlo
# En Windows:
venv\Scripts\activate
# En Mac/Linux:
source venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt
```

## Correr la app

```bash
uvicorn app:app --reload
```

Abrí el browser en: **http://localhost:8000**

## Uso

1. Ingresá tu **Gemini API Key** y hacé click en **Verificar**
2. Subí el video del partido
3. En el primer frame, **dibujá un rectángulo** sobre el jugador que querés analizar
4. Hacé click en **"Analizar jugador"** y esperá
5. Cuando termina, podés **exportar el Excel** con el resumen y timeline

## Tips para mejores resultados

- Videos donde el jugador sea **bien visible** (no cortado por el borde)
- Encuadre del rectángulo **ajustado al cuerpo** del jugador
- Videos con **buena resolución** y sin mucho motion blur
