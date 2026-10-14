# Arquivo Científico e Histórico de Experimentos

Este diretório preserva a trajetória de pesquisa e experimentação empírica que culminou no desenvolvimento da arquitetura **VQ2** e **VQ2M (Paged Streaming MoE)**.

---

## 1. Estrutura do Diretório

- **`results/`**: Arquivos `.json` com métricas consolidadas de perplexidade, divergência KL, concordância top-1, sensibilidade Hessiana e distorção Rate-Distortion (RDO) coletadas em hardware real (NVIDIA GeForce RTX 3060 12GB).
- **`legacy/`**: Scripts de experimentação histórica organizados pelas fases cronológicas da pesquisa.

---

## 2. As Fases da Pesquisa

### Fase 1: Diagnóstico e Limitações do H5-A (`h5a_*`)
* **Problema investigado**: O método H5-A anterior atingia colapso funcional ao tentar compactar modelos abaixo de 850 MB.
* **Descobertas**:
  1. A ilusão da métrica: avaliar em poucas frases curtas gerava sobreajuste dos adaptadores.
  2. Erro de Frobenius vs Erro Funcional: quantizar sem compensação Hessiana destruía canais de alta sensibilidade.
  3. Matrizes sensíveis a outliers em blocos de 8 dimensões sem rotação prévia.

### Fase 2: Compensação Hessiana LDLQ e Rotação RHT (`pilot_h5a_hessian.py`, `run_deltanet_*`)
* **Inovação**: Introdução da **Randomized Hadamard Transform (RHT)** para eliminação de *activation outliers* e ortogonalização isotrópica.
* **Resultado**: Formulação do algoritmo **LDLQ** (Cholesky inversa amortecida da Hessiana $H = E[XX^T]$), permitindo anular distorções de 2ª ordem durante o empacotamento vetorial.

### Fase 3: Rate-Distortion Optimization (RDO) (`sweep_*.py`, `sensitivity_*.py`)
* **Inovação**: Análise de sensibilidade funcional por camada ($\Delta\text{Distortion}/\Delta\text{MB}$).
* **Descoberta**: Camadas de embedding e FFN suportam 2.0 bpw sem impacto cognitivo, enquanto projeções de atenção (`k_proj`, `v_proj`, `out_proj`) exigem 3.0 bpw.
* **Resultado**: Redução do modelo 2B para 560 MB / 678 MB com PPL comparável a Q4_K_M.

### Fase 4: Desacoplamento de Memória MoE & Streaming Paginado (VQ2M)
* **Inovação**: Mapeamento de memória `mmap` com buffers de camada e cache LRU.
* **Resultado**: Execução do modelo MoE de 35.5B parâmetros em menos de 1.25 GB de VRAM ativa.
