import os
import glob
from pathlib import Path

def eliminar_pdfs_de_carpetas():
    """
    Script para eliminar archivos PDF de 6 carpetas específicas
    """
    
    # CONFIGURA AQUÍ LAS RUTAS DE TUS 6 CARPETAS
    carpetas = [
        r"/home/turing/Documents/gsgp-visual/Visualizing-Semantic-Space-Exploration-in-GSGP/figures/umap/population_64/gsm/trace/concrete_new_plots",
        r"/home/turing/Documents/gsgp-visual/Visualizing-Semantic-Space-Exploration-in-GSGP/figures/umap/population_64/gsm/trace/Ecoling_new_plots", 
        r"/home/turing/Documents/gsgp-visual/Visualizing-Semantic-Space-Exploration-in-GSGP/figures/umap/population_64/gsm/trace/Eheating_new_plots",
        r"/home/turing/Documents/gsgp-visual/Visualizing-Semantic-Space-Exploration-in-GSGP/figures/umap/population_64/gsm/trace/Housing_new_plots",
        r"/home/turing/Documents/gsgp-visual/Visualizing-Semantic-Space-Exploration-in-GSGP/figures/umap/population_64/gsm/trace/Tower_new_plots",
        r"/home/turing/Documents/gsgp-visual/Visualizing-Semantic-Space-Exploration-in-GSGP/figures/umap/population_64/gsm/trace/yatch_new_plots"
        
    ]
    
    total_eliminados = 0
    errores = []
    
    print("=== ELIMINADOR DE ARCHIVOS PDF ===\n")
    
    # Mostrar carpetas a procesar
    print("Carpetas a procesar:")
    for i, carpeta in enumerate(carpetas, 1):
        print(f"{i}. {carpeta}")
    
    # Confirmación de seguridad
    confirmacion = input("\n¿Estás seguro de que quieres eliminar todos los archivos PDF de estas carpetas? (si/no): ")
    
    if confirmacion.lower() not in ['si', 'sí', 'yes', 'y', 's']:
        print("Operación cancelada.")
        return
    
    # Procesar cada carpeta
    for carpeta in carpetas:
        print(f"\nProcesando carpeta: {carpeta}")
        
        # Verificar si la carpeta existe
        if not os.path.exists(carpeta):
            print(f"  ⚠️  La carpeta no existe: {carpeta}")
            errores.append(f"Carpeta no encontrada: {carpeta}")
            continue
        
        # Buscar archivos PDF en la carpeta
        patron_pdf = os.path.join(carpeta, "*.pdf")
        archivos_pdf = glob.glob(patron_pdf)
        
        if not archivos_pdf:
            print(f"  ℹ️  No se encontraron archivos PDF")
            continue
        
        print(f"  📄 Encontrados {len(archivos_pdf)} archivos PDF")
        
        # Eliminar cada archivo PDF
        eliminados_carpeta = 0
        for archivo_pdf in archivos_pdf:
            try:
                nombre_archivo = os.path.basename(archivo_pdf)
                os.remove(archivo_pdf)
                print(f"    ✅ Eliminado: {nombre_archivo}")
                eliminados_carpeta += 1
                total_eliminados += 1
            except Exception as e:
                error_msg = f"Error al eliminar {archivo_pdf}: {str(e)}"
                print(f"    ❌ {error_msg}")
                errores.append(error_msg)
        
        print(f"  📊 Eliminados {eliminados_carpeta} archivos de esta carpeta")
    
    # Resumen final
    print(f"\n=== RESUMEN ===")
    print(f"Total de archivos PDF eliminados: {total_eliminados}")
    
    if errores:
        print(f"\nErrores encontrados ({len(errores)}):")
        for error in errores:
            print(f"  - {error}")
    else:
        print("✅ Proceso completado sin errores")

def mostrar_pdfs_sin_eliminar():
    """
    Función para solo mostrar los PDFs que se encontrarían, sin eliminarlos
    """
    carpetas = [
        r"/home/turing/Documents/gsgp-visual/Visualizing-Semantic-Space-Exploration-in-GSGP/figures/umap/population_64/gsm/trace/concrete_new_plots",
        r"/home/turing/Documents/gsgp-visual/Visualizing-Semantic-Space-Exploration-in-GSGP/figures/umap/population_64/gsm/trace/Ecoling_new_plots", 
        r"/home/turing/Documents/gsgp-visual/Visualizing-Semantic-Space-Exploration-in-GSGP/figures/umap/population_64/gsm/trace/Eheating_new_plots",
        r"/home/turing/Documents/gsgp-visual/Visualizing-Semantic-Space-Exploration-in-GSGP/figures/umap/population_64/gsm/trace/Housing_new_plots",
        r"/home/turing/Documents/gsgp-visual/Visualizing-Semantic-Space-Exploration-in-GSGP/figures/umap/population_64/gsm/trace/Tower_new_plots",
        r"/home/turing/Documents/gsgp-visual/Visualizing-Semantic-Space-Exploration-in-GSGP/figures/umap/population_64/gsm/trace/yatch_new_plots"
        
    ]
    
    
    print("=== VISTA PREVIA - ARCHIVOS PDF ENCONTRADOS ===\n")
    
    total_pdfs = 0
    
    for carpeta in carpetas:
        print(f"Carpeta: {carpeta}")
        
        if not os.path.exists(carpeta):
            print(f"  ⚠️  La carpeta no existe")
            continue
        
        patron_pdf = os.path.join(carpeta, "*.pdf")
        archivos_pdf = glob.glob(patron_pdf)
        
        if not archivos_pdf:
            print(f"  ℹ️  No se encontraron archivos PDF")
        else:
            print(f"  📄 {len(archivos_pdf)} archivos PDF encontrados:")
            for archivo in archivos_pdf:
                nombre = os.path.basename(archivo)
                print(f"    - {nombre}")
            total_pdfs += len(archivos_pdf)
        print()
    
    print(f"Total de archivos PDF que se eliminarían: {total_pdfs}")

if __name__ == "__main__":
    print("Selecciona una opción:")
    print("1. Ver archivos PDF (sin eliminar)")
    print("2. Eliminar archivos PDF")
    
    opcion = input("\nIngresa tu opción (1 o 2): ")
    
    if opcion == "1":
        mostrar_pdfs_sin_eliminar()
    elif opcion == "2":
        eliminar_pdfs_de_carpetas()
    else:
        print("Opción no válida")
