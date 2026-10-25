# Demostraciones de Machine Learning (Jupyter Book)

Jupyter Book con las demostraciones de la tarea: k-NN, regresión Ridge, regresión Lasso, ejercicio 6 de árboles (Gini) y AdaBoost.

## Publicar en GitHub Pages

1. Crea un repositorio en GitHub y sube todo el contenido de esta carpeta a la rama `main`.
2. En `book/_config.yml` cambia `https://github.com/USUARIO/REPOSITORIO` por la URL de tu repositorio.
3. En GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
4. Cada `push` a `main` compila y publica el libro (`.github/workflows/deploy.yml`).

## Compilar localmente

```bash
pip install -r requirements.txt
jupyter-book build book/
```

## Estructura

- `book/`: capítulos en MyST Markdown, `_config.yml`, `_toc.yml` y `references.bib`.
- `verificacion/`: scripts de verificación numérica.
- `.github/workflows/deploy.yml`: compilación y despliegue automáticos.
