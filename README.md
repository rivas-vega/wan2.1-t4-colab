<div align="center">
  <img src="assets/hero_banner.svg" alt="Wan 2.1 Video Studio" width="100%" />
  <br/><br/>
  <a href="https://colab.research.google.com/github/rivas-vega/wan2.1-t4-colab/blob/main/wan2_1_t4_colab_free_inference.ipynb">
    <img src="https://img.shields.io/badge/Launch_in_Colab-Google_Colab_T4-0064e0?style=flat-square&logo=googlecolab&logoColor=white" alt="Open In Colab" />
  </a>
  <a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers">
    <img src="https://img.shields.io/badge/Weights-Wan_2.1_T2V_1.3B-f3f4f6?style=flat-square&labelColor=111827&color=374151" alt="Model Weights" />
  </a>
  <img src="https://img.shields.io/badge/Hardware-NVIDIA_T4_(15GB_VRAM_%2F_12.7GB_RAM)-f3f4f6?style=flat-square&labelColor=111827&color=374151" alt="Hardware" />
  <img src="https://img.shields.io/badge/Precision-4--bit_NF4_UMT5_%2B_FP16_DiT_%2B_FP32_VAE-f3f4f6?style=flat-square&labelColor=111827&color=374151" alt="Precision" />
</div>

<br/>

## Overview

Alibaba's **Wan 2.1 (1.3B)** is an open-weights text-to-video foundation model composed of:
1. **UMT5-XXL** (text encoder, 22.7 GB in FP32)
2. **Wan 2.1 DiT** (1.3B diffusion transformer)
3. **3D Causal VAE** (spatiotemporal video autoencoder)

Running this pipeline directly with default Hugging Face `diffusers` requires 32+ GB of system RAM and 24+ GB of VRAM. On a free **Google Colab Tesla T4** (`12.7 GB System RAM`, `15.0 GB VRAM`), the environment crashes during text encoder loading (RAM OOM) or during final 3D VAE decoding (CUDA OOM).

This repository provides an inference pipeline and minimalist Gradio web studio engineered to run Wan 2.1 on free Google Colab hardware, generating continuous **4.1-second (65 frames)** and **5.1-second (81 frames)** videos at 480p.

---

## System Architecture & Demo

<div align="center">
  <img src="assets/pipeline_architecture.svg" alt="Wan 2.1 Inference Pipeline" width="100%" />
</div>

<br/>

<table>
  <tr>
    <td width="42%" align="center" valign="middle">
      <img src="assets/demo_t4_vlog.gif" alt="Generated sample on T4 GPU" width="300" style="border-radius: 12px;" />
    </td>
    <td width="58%" valign="middle">
      <h4>Unedited Output on Tesla T4</h4>
      <p><b>Prompt used:</b></p>
      <blockquote>
        <i>"A trendy young content creator holding a small vlogging camera at arm's length, walking through a sunny historic European street, smiling naturally to the camera with soft daylight and vibrant cafe terraces in the background"</i>
      </blockquote>
      <p>
        <b>Specs:</b><br/>
        • <b>Resolution:</b> 480 × 832 (9:16 vertical)<br/>
        • <b>Length:</b> 65 frames · 4.1 s @ 16 fps<br/>
        • <b>Scheduler:</b> UniPC Multistep (45 steps, <code>flow_shift = 3.0</code>, <code>CFG = 5.0</code>)<br/>
        • <b>Seed:</b> <code>342683004</code><br/>
        • <b>GPU:</b> NVIDIA Tesla T4 (Google Colab Free Tier)
      </p>
    </td>
  </tr>
</table>

---

## Techniques & Precision Strategy

### ¿Usamos cuantización? ¿Qué se ha cuantizado y qué no?

| Componente | Tamaño Original | Precisión Utilizada | ¿Cuantizado? | Dónde reside | Uso de Memoria |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Text Encoder (`UMT5-XXL`)** | 22.7 GB | **4-bit NF4** (`bitsandbytes`) | **Sí** | GPU VRAM | **2.4 GB VRAM** (RAM a 3.0 GB) |
| **DiT (`WanTransformer3DModel`)** | 2.6 GB | **FP16 Nativo** | **No** | GPU VRAM (pasa a RAM pre-VAE) | **2.6 GB VRAM** |
| **3D Causal VAE** | 1.2 GB | **FP32** (con tiling espacial/temporal) | **No** | GPU VRAM | **~6.5 GB VRAM** en decodificación |

<br/>

### 1. Cuantización 4-bit NF4 directa a GPU para UMT5
- **El problema:** `UMT5-XXL` pesa 22.7 GB en FP32 repartido en 6 shards. Cargarlo en la CPU o usar `device_map="auto"` desborda de inmediato los 12.7 GB de RAM de Colab y el kernel de Linux mata el proceso.
- **La solución:** Descargamos los shards directamente a la GPU (`device_map={"": 0}`) usando cuantización NormalFloat4 (`bnb_4bit_quant_type="nf4"`). Esto comprime el modelo a **2.4 GB de VRAM** y mantiene la RAM del sistema en apenas **3.0 GB**, dejando casi 10 GB de RAM libres.

### 2. DiT en FP16 nativo (sin cuantizar)
- El transformer de difusión de 1.3B parámetros se mantiene en `float16` nativo (~2.6 GB en VRAM).
- **Por qué no cuantizarlo:** La cuantización a 4 u 8 bits en el DiT degrada la coherencia temporal y produce pérdida de microtextura (piel, cabello, telas). Mantenerlo en FP16 preserva toda la nitidez visual y aprovecha al 100% los Tensor Cores de la GPU T4.

### 3. VAE 3D en FP32 con Tiling
- El VAE causal 3D corre en `float32` con `enable_tiling()` y `enable_slicing()`.
- En Wan 2.1, ejecutar el VAE en FP16 genera posterización y artefactos de color (color banding). Mantenerlo en FP32 garantiza gradientes limpios y colores naturales sin superar el límite de memoria gracias a la división en bloques espaciales y temporales.

### 4. Desalojo dinámico pre-VAE (Pre-VAE CPU Eviction)
- **El problema:** Decodificar 65 fotogramas de vídeo 3D en FP32 requiere un pico de memoria considerable. Si el DiT (2.6 GB) permanece en la GPU mientras el VAE decodifica, la VRAM total supera los 15.0 GB y ocurre un fallo CUDA OOM.
- **La solución:** Al finalizar el último paso de difusión (`step_index == total_steps - 1`), la pipeline traslada temporalmente `pipe.transformer` a la RAM del sistema (`transformer.to("cpu")`) y libera la memoria de CUDA con `torch.cuda.empty_cache()`. Esto otorga más de 4.5 GB de VRAM adicionales de margen para la decodificación del vídeo.

### 5. Calibración de ruido para 480p (`flow_shift = 3.0`)
- Por defecto, la configuración de Wan 2.1 utiliza `flow_shift = 5.0`, optimizada para 720p. En resoluciones de 480p ($480 \times 832$), este valor concentra demasiados pasos en la fase de alto ruido y resta iteraciones a la fase de bajo ruido donde se forman las texturas.
- Usar **`flow_shift = 3.0`** con el solver `UniPCMultistepScheduler` (45 pasos) equilibra la trayectoria y recupera el detalle en la imagen final.

### 6. Optimización de memoria a nivel de sistema
- `HF_HUB_DISABLE_XET=1`: Desactiva la reconstrucción en RAM de archivos descargados de Hugging Face.
- `glibc malloc_trim(0)`: Invoca la liberación forzada del asignador de memoria C para devolver los buffers liberados por PyTorch al sistema operativo inmediatamente.

---

## Comparativa de Memoria

<div align="center">
  <img src="assets/memory_telemetry.svg" alt="Memory Telemetry" width="100%" />
</div>

<br/>

| Escenario | RAM del Sistema (Límite: 12.7 GB) | VRAM de la GPU (Límite: 15.0 GB) | Resultado |
| :--- | :---: | :---: | :---: |
| **Diffusers estándar** | `13.9 GB` | `> 15.0 GB` | **Crash (OOM)** al cargar UMT5 o en VAE |
| **Esta pipeline optimizada** | **`3.0 GB`** *(9.7 GB libres)* | **`10.4 GB`** *(4.6 GB libres)* | **Generación exitosa** (65–81 frames) |

---

## Presets de Generación

| Formato | Resolución | Duración | Frames | Configuración del Solver | VRAM Pico |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Reel 9:16 · 4s`** *(Recomendado)* | `480 × 832` | **4.1 s** (16 fps) | `65` | UniPC (flow_shift=3.0, 45 steps, CFG=5.0) | `10.4 GB` |
| **`Reel 9:16 · 5s`** | `480 × 832` | **5.1 s** (16 fps) | `81` | UniPC (flow_shift=3.0, 45 steps, CFG=5.0) | `11.2 GB` |
| **`Cinema 16:9 · 4s`** | `832 × 480` | **4.1 s** (16 fps) | `65` | UniPC (flow_shift=3.0, 45 steps, CFG=5.0) | `10.4 GB` |

---

## Cómo Ejecutar en Google Colab

1. Abre [`wan2_1_t4_colab_free_inference.ipynb`](https://colab.research.google.com/github/rivas-vega/wan2.1-t4-colab/blob/main/wan2_1_t4_colab_free_inference.ipynb) en Google Colab.
2. Selecciona **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4**.
3. Ejecuta las celdas en orden:
   - **Paso 1:** Instala dependencias y configura variables de entorno.
   - **Paso 2:** *(Opcional)* Token de Hugging Face para acelerar la descarga.
   - **Paso 3 / 4:** Inicia el estudio interactivo con interfaz Gradio limpia.

---

## Estructura del Repositorio

```text
├── assets/
│   ├── hero_banner.svg             # Banner de cabecera minimalista
│   ├── pipeline_architecture.svg   # Diagrama de flujo del pipeline
│   ├── memory_telemetry.svg        # Comparativa de uso de RAM/VRAM
│   ├── demo_t4_vlog.gif            # GIF de muestra con bordes redondeados
│   └── demo_t4_vlog.mp4            # Vídeo original H.264 generado en T4
├── wan2_1_t4_colab_free_inference.ipynb  # Notebook listo para Colab
├── app.py                          # Interfaz web de Gradio limpia
├── pipeline_utils.py               # Loader 4-bit NF4, desalojo pre-VAE y scheduler 480p
├── requirements.txt                # Dependencias de Python
└── README.md                       # Documentación técnica
```

---

## Autor

**Pedro** ([@rivas-vega](https://github.com/rivas-vega))  
M2 MVA, École Normale Supérieure Paris-Saclay · Grado en Matemáticas, Universidad de Sevilla
