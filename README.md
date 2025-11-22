# Modelo 3D / 2D - Relieve Marcona

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)

## 📋 Descripción

Este proyecto genera y visualiza un relieve 3D procedimental de la zona de Marcona utilizando **Python**, **Pygame** y **PyOpenGL**. Permite una exploración interactiva del terreno generado, ofreciendo una herramienta visual para el análisis topográfico simulado.

## ✨ Características

- **Generación Procedimental**: Creación de terrenos 3D dinámicos.
- **Visualización Interactiva**: Navegación en tiempo real sobre el modelo 3D.
- **Renderizado Eficiente**: Uso de OpenGL para un rendimiento óptimo.
- **Modo 2D/3D**: Capacidad de visualización en diferentes perspectivas (según implementación).

## 🚀 Instalación

Sigue estos pasos para configurar el entorno de desarrollo:

### Prerrequisitos
- Python 3.8 o superior.

### Configuración del Entorno

1.  **Clonar el repositorio** (si aún no lo has hecho):
    ```bash
    git clone git@github.com:DANSOBeron0/MODELO-3D-2D-MARCONA-.git
    cd MODELO-3D-2D-MARCONA-
    ```

2.  **Crear y activar un entorno virtual**:
    ```powershell
    # Windows (PowerShell)
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

3.  **Instalar dependencias**:
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Uso

Para iniciar la visualización del modelo 3D, ejecuta el script principal:

```bash
python relieve_3d.py
```

## 📂 Estructura del Proyecto

```text
.
├── src/                # Código fuente adicional
├── relieve.py          # Script de generación de relieve base
├── relieve_3d.py       # Script principal de visualización 3D
├── requirements.txt    # Lista de dependencias del proyecto
├── .gitignore          # Archivos ignorados por Git
├── LICENSE             # Licencia del proyecto
└── README.md           # Documentación del proyecto
```

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

## 👤 Autor

Desarrollado por [DANSOBeron0](https://github.com/DANSOBeron0).
