import os
import cv2
import numpy as np
from ultralytics import YOLO
# --- NUEVAS LIBRERÍAS PARA MEDICIÓN PRECISA ---
from skimage.morphology import skeletonize
from skimage import io, color
import math
# ---------------------------------------------

# =========================================================================
# CONFIGURACIÓN INICIAL (AJUSTA ESTAS RUTAS)
# =========================================================================
# CAMBIA ESTA RUTA: Debe apuntar al directorio raíz de tu dataset de Roboflow 
# exportado en formato YOLOv8 Segmentation. ¡VERIFICA QUE ESTÉ CORRECTA!
ROBOFLOW_DATA_PATH = r'C:\Users\ferno\Desktop\Proyecto computacional\Arabidopsis Primary Root' 

# Modelo inicial ligero para comenzar el entrenamiento
MODEL_NAME = 'yolov8n-seg.pt' 
# Ruta esperada del modelo final después del entrenamiento (no necesita ser editada)
OUTPUT_MODEL_PATH = 'runs/segment/train/weights/best.pt' 
# Ruta donde se guardarán las imágenes de prueba procesadas (no necesita ser editada)
OUTPUT_TEST_IMAGE_PATH = 'runs/segment/processed_tests/'


# =========================================================================
# 1. FUNCIÓN DE CALIBRACIÓN Y MEDICIÓN (CON ESQUELETIZACIÓN)
# =========================================================================

def calculate_length(mask_image, scale_cm_per_pixel):
    """
    Función para medir la longitud de la raíz principal utilizando esqueletización.
    
    CRÍTICO: Este método es mucho más preciso que el cálculo de área simple.
    Convierte la máscara 2D en una línea de 1 píxel de ancho (esqueleto) 
    y cuenta la longitud en píxeles.
    """
    
    if mask_image is None or np.sum(mask_image) == 0:
        return 0.0, 0
    
    # Asegurar que la máscara esté en escala de grises y binaria (True/False)
    if mask_image.ndim == 3:
        mask_binary = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    else:
        mask_binary = mask_image
        
    # El esqueletizado requiere valores booleanos (True para raíz, False para fondo)
    mask_bool = (mask_binary > 0)
        
    # Aplicar la esqueletización
    # Esto reduce la máscara a su línea central
    skeleton = skeletonize(mask_bool)
    
    # Contar la longitud en píxeles del esqueleto
    estimated_length_pixels = np.sum(skeleton)
    
    # Aplicar la calibración a CM
    length_cm = estimated_length_pixels * scale_cm_per_pixel
    
    return length_cm, estimated_length_pixels

# =========================================================================
# 2. FUNCIÓN DE ENTRENAMIENTO INICIAL (Paso 4)
# =========================================================================

def train_model():
    """Configura y ejecuta el entrenamiento del modelo YOLOv8-seg."""
    print("--- Comenzando el Entrenamiento Inicial ---")
    print("Nota: El entrenamiento puede tardar varios minutos/horas dependiendo del hardware.")
    
    # Cargar el modelo pre-entrenado para segmentación
    model = YOLO(MODEL_NAME)
    
    # Configuración de entrenamiento con hiperparámetros clave
    print(f"Entrenando con {MODEL_NAME} en {ROBOFLOW_DATA_PATH}")
    results = model.train(
        data=os.path.join(ROBOFLOW_DATA_PATH, 'data.yaml'),
        epochs=75,             # Número de épocas (AJUSTAR: 75 es un buen inicio)
        imgsz=640,             # Tamaño de imagen recomendado para YOLOv8
        batch=4,               # Tamaño de lote pequeño (AJUSTAR a 8 o 16 si tienes buena GPU)
        name='arabidopsis_root_segmentation_v1', 
        patience=15,           # Detener si no hay mejora en validación después de 15 épocas
        workers=8              # Número de workers para cargar datos (ajustar a 4, 8, etc.)
    )
    
    print("--- Entrenamiento Finalizado. Resultados guardados en 'runs/segment/train' ---")
    return results

# =========================================================================
# 3. FUNCIÓN DE EVALUACIÓN (Paso 6)
# =========================================================================

def evaluate_model():
    """Mide la precisión del modelo en el conjunto de prueba y genera métricas."""
    print("--- Evaluando el Modelo en el Conjunto de Prueba ---")
    
    if not os.path.exists(OUTPUT_MODEL_PATH):
        print(f"ERROR: El modelo final '{OUTPUT_MODEL_PATH}' no se encontró. Entrena el modelo primero.")
        return None
        
    model = YOLO(OUTPUT_MODEL_PATH)
    
    # Realizar la evaluación en el conjunto 'test' (15% de tus datos)
    metrics = model.val(
        data=os.path.join(ROBOFLOW_DATA_PATH, 'data.yaml'),
        split='test' # Evaluar específicamente en el conjunto de prueba
    )
    
    # Muestra de métricas (mAP = mean Average Precision, IoU = Intersection over Union)
    print("\n----------------------------------------------------")
    print(f"METRICAS DE EVALUACIÓN EN EL CONJUNTO DE PRUEBA:")
    print(f"mAP50 (Caja Delimitadora): {metrics.box.map50:.4f}")         
    print(f"mAP50-95 (Segmentación/IoU): {metrics.seg.map:.4f}") 
    print("----------------------------------------------------")
    
    # La inspección visual se realiza en 'runs/segment/val' donde se guardan las predicciones.
    return metrics

# =========================================================================
# 4. PRUEBAS PRÁCTICAS Y MEDICIÓN (Paso 7)
# =========================================================================

def process_new_image(image_path, scale_cm_per_pixel):
    """Realiza la predicción en una nueva imagen no vista y calcula la longitud."""
    print(f"\n--- Procesando Nueva Imagen de Prueba: {image_path} ---")
    
    if not os.path.exists(OUTPUT_MODEL_PATH):
        print(f"ERROR: El modelo final '{OUTPUT_MODEL_PATH}' no se encontró. Entrena el modelo primero.")
        return

    # Cargar el modelo final entrenado
    model = YOLO(OUTPUT_MODEL_PATH)
    
    try:
        # Realizar predicción. Usamos un umbral de confianza alto para solo tomar la raíz principal
        results = model(image_path, save=False, save_conf=False, conf=0.7, iou=0.5, verbose=False)
        result = results[0] # Obtener el primer resultado 
        
        if result.masks is None or len(result.masks.data) == 0:
            print("FALLO: No se detectó la raíz principal en la imagen.")
            return

        # Elige la máscara con mayor confianza (asumiendo que es la raíz principal)
        confidences = result.boxes.conf.cpu().tolist()
        best_index = confidences.index(max(confidences))
        
        # Obtener la máscara de la mejor detección
        best_mask_data = result.masks.data[best_index].cpu().numpy().astype(np.uint8) 
        
        # Redimensionar la máscara a la resolución original de la imagen
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"ERROR: No se pudo cargar la imagen en {image_path}. Verifica la ruta.")
            return
            
        mask_resized = cv2.resize(best_mask_data, 
                                  (original_img.shape[1], original_img.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST) * 255
        
        mask_color_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        
        # 1. Medición de la Longitud
        root_length_cm, root_length_pixels = calculate_length(mask_resized, scale_cm_per_pixel)
        
        # 2. Inspección Visual
        
        # Usar la función de trazado de YOLOv8 para visualización
        annotated_img = result.plot()
        
        # Dibujar el esqueleto sobre la imagen anotada (Opcional, pero útil para visualizar)
        skeleton_bool = skeletonize((mask_resized > 0))
        skeleton_bgr = (np.stack([skeleton_bool * 255] * 3, axis=-1)).astype(np.uint8)
        
        # Mezclar el esqueleto (azul) con la imagen anotada
        # Convertir a flotante, sumar y normalizar
        combined_img = cv2.addWeighted(annotated_img, 0.8, skeleton_bgr, 0.2, 0)


        # Añadir el resultado de la medición al gráfico
        text_to_display = f"Longitud (Esqueleto): {root_length_cm:.2f} cm"
        cv2.putText(
            combined_img, 
            text_to_display, 
            (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, 
            (0, 255, 0), # Verde
            2
        )
        
        # Guardar la imagen de prueba procesada
        os.makedirs(OUTPUT_TEST_IMAGE_PATH, exist_ok=True)
        output_filename = os.path.join(OUTPUT_TEST_IMAGE_PATH, 'processed_' + os.path.basename(image_path))
        cv2.imwrite(output_filename, combined_img)
        
        print(f"Longitud estimada de la raíz principal: {root_length_cm:.2f} cm ({root_length_pixels} píxeles)")
        print(f"Resultado visual guardado en: {output_filename}")
        
    except Exception as e:
        print(f"Ocurrió un error al procesar la imagen. Verifica que la clase 'primary_root' exista en tu modelo y que la imagen cargue correctamente. Error: {e}")
        # print(traceback.format_exc()) # Descomentar para debug
        

# =========================================================================
# FUNCIÓN PRINCIPAL DE EJECUCIÓN
# =========================================================================

def main():
    
    # --- CALIBRACIÓN DE ESCALA (CLAVE PARA LA MEDICIÓN) ---
    # AJUSTA ESTE VALOR: Mide cuántos píxeles tiene 1 cm de la regla en tus fotos.
    # Fórmula: CM_PER_PIXEL_SCALE = 1 / (Píxeles por 1 cm)
    # Ejemplo: Si 1 cm son 200 píxeles, usa 0.005.
    CM_PER_PIXEL_SCALE = 0.005 
    print(f"** Escala de Calibración Usada: {CM_PER_PIXEL_SCALE} cm/pixel **")
    
    # --- PASO 1: ENTRENAMIENTO ---
    # COMENTA esta línea si ya entrenaste el modelo y tienes el archivo 'best.pt'
    train_model() 
    
    # --- PASO 2: EVALUACIÓN ---
    evaluate_model()
    
    # --- PASO 3: PRUEBAS PRÁCTICAS ---
    # CAMBIA ESTA RUTA: Usa una imagen que NO esté en los conjuntos de train/val/test
    NEW_IMAGE_FOR_TESTING = 'C:\\Users\\ferno\\Desktop\\Proyecto computacional\\Arabidopsis Primary Root\\data\\Control_1.jpg' 
    
    if os.path.exists(NEW_IMAGE_FOR_TESTING):
        process_new_image(NEW_IMAGE_FOR_TESTING, CM_PER_PIXEL_SCALE)
    else:
        print(f"\nADVERTENCIA: Archivo de prueba no encontrado en: {NEW_IMAGE_FOR_TESTING}")
        print("Asegúrate de: 1) Cambiar la ruta de 'NEW_IMAGE_FOR_TESTING' en la función 'main()'. 2) Usar una imagen fuera de tu dataset de entrenamiento.")
    
    print("\n--- Pipeline Finalizado ---")

if __name__ == "__main__":
    main()
