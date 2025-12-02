# Resumen del Dockerfile principal

1.- **Base**: se utiliza 'python:3.11-slim' para mantener la imagen ligera
2.- **Dependencias de sistema**:se instalan las herramientas de compilación mínimas,(build-essential, gcc / g++, curl y git) y se limpia el caché de 'apt' para reducir el tamaño de la capa.
3.- **Entorno de Python**: Se definen variables para evitar archivos '.pyc' y habilitar salida sin buffer. 'pip' se actualiza antes de instalar 'requirements.txt', sin usar el cache.
4.- **Código de la aplicación**: se copia el resto del repositorio en '/app' y se expone el puerto 8080 para servicios locales.
5.- **Ejecución**: el comando de inicio sincroniza datos con dvc si está disponible y ejecuta 'kedro run' con el pipeline indicado en 'KEDRO_PIPELINE' (por defecto '__default__').

Estas etapas ayudan el cacheo de dependencias y dejan clara la logica de construcción para futuras optimizaciones :D