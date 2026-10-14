# 🧭 Guia para Pesquisadores: Próximo Marco Central — LEGION CPU-First
**Roteiro Técnico, Especificação de Tarefas e O que Falta para Execução Real em PCs Básicos e Android**

---

## 1. O Diagnóstico: O Que Já Está Contemplado e O Que Ainda Falta

A visão central do projeto é clara: **o LEGION deve funcionar de verdade em computadores básicos e celulares Android, sem depender de uma GPU dedicada.**

O repositório já documenta com solidez os fundamentos matemáticos e os pipelines acelerados por GPU:
* **Formato VQ2 e Modelo B2 (Documentado)**: Compressão vetorial, formato paginado `.vq2m`, índice de 10.240 especialistas e cache LRU fazem parte da arquitetura documentada.
* **Backend CUDA (Implementado e Validado)**: Kernels Triton, streaming MoE e CUDA Graphs formam o caminho acelerado para GPU NVIDIA.

Contudo, para atingir o objetivo de IA portátil em hardware modesto, a **implementação CPU-First** precisa ser tratada como uma etapa prioritária própria:

| Componente | Status Atual | Desafio Técnico Central |
| :--- | :---: | :--- |
| **Projeções Densas Comprimidas na CPU** | 🔥 **Prioridade Máxima** | É o gargalo revelado pelas medições: **328 matrizes densas** estão sendo materializadas em BF16 na RAM. Precisamos de execução CPU que evite manter todas essas matrizes reconstruídas. |
| **Kernels Nativos AVX2 e NEON** | 📋 **Planejado** | O roadmap cita SIMD para CPU x86_64 e ARM NEON, mas é necessário transformar essa intenção em kernels C++/SIMD testados e integrados ao runtime. |
| **Aplicativo Android** | 📋 **Planejado** | Definir o caminho real de execução do formato VQ2/VQ2M (via CPU ARM64 / NPU / OpenCL) e sua integração com o LEGION. |

---

## 2. A Peça Mais Importante que Falta: Um Backend CPU de Verdade

Hoje, o desafio não é simplesmente fazer o PyTorch aceitar `device="cpu"`. É conseguir que as operações matemáticas **consumam os pesos comprimidos sem exigir a reconstrução de todas as matrizes em BF16**.

As tarefas prioritárias do projeto estão estruturadas nesta ordem de execução:

```
[CPU-01: Projeções sem Materialização Permanente]
                      │
                      ▼
[CPU-02: Kernels SIMD Nativos (AVX2 -> ARM NEON)]
                      │
                      ▼
[CPU-03: Orçamento Adaptativo de Memória Global]
                      │
                      ▼
[CPU-04: Benchmark Rigoroso e Reproduzível]
                      │
                      ▼
[MOBILE-01: Portabilidade Android ARM64]
```

### 🎯 CPU-01 — Projeções VQ2 sem Materialização Permanente
* **Objetivo**: Identificar as **328 matrizes densas** (atenções lineares DeltaNet, projeções QKV, projeções de saída e shared expert) e implementar `quant_linear` para CPU diretamente sobre a representação compactada (índices `uint16`/`uint8` + escalas + codebooks de 8 dimensões).
* **Mecanismo**: Desquantização sob demanda em buffers temporários limitados por thread, liberados imediatamente após o produto interno, impedindo que gigabytes de pesos permaneçam alocados na RAM.

### ⚡ CPU-02 — Kernels SIMD Nativos (AVX2 & ARM NEON)
* **Objetivo**: Desenvolver desquantizadores vetoriais diretos em C++/SIMD para substituir loops interpretados em Python.
* **Passo 1**: Implementar referência escalar em C++ para verificação estrita de correção numérica contra o PyTorch.
* **Passo 2**: Vetorização AVX2 (256 bits) para PCs convencionais x86_64 com lookup e FMA paralelo em registradores YMM.
* **Passo 3**: Portar a mesma formulação matemática para ARM NEON (128 bits) em registradores Q para execução eficiente em processadores ARM64 (Snapdragon, MediaTek, Apple Silicon).

### 🛡️ CPU-03 — Orçamento Adaptativo de Memória
* **Objetivo**: Estabelecer um teto rígido de consumo de RAM do processo.
* **Abrangência**: Limitar não apenas o cache MoE de especialistas, mas também:
  - Buffers de projeção temporários.
  - Matrizes intermediárias de ativação.
  - Memória de contexto (KV-Cache estático/rolling).
  - Alocações do runtime e da biblioteca de tensores.

### 📊 CPU-04 — Benchmark Rigoroso e Reproduzível
* **Objetivo**: Registrar métricas empíricas comparativas antes e depois de cada otimização sob a mesma sequência de entrada:
  - Tempo de carga fria (cold boot).
  - Geração aquecida (tokens/segundo).
  - RSS de pico da RAM do sistema.
  - Latência Time-To-First-Token (TTFT) e latência P95 por token.
  - Taxa de acerto (hit rate) do cache LRU de especialistas.
  - Fidelidade numérica (similaridade cosseno e divergência KL contra BF16 original).

### 📱 MOBILE-01 — Portabilidade Android
* **Objetivo**: Validar a execução ponta a ponta em dispositivos móveis Android.
* **Requisitos**:
  - Leitura sob demanda do modelo `.vq2m` diretamente do armazenamento interno UFS via `mmap`.
  - Execução contida dentro da memória disponível de aparelhos com 4 GB a 8 GB de RAM total.
  - Compilação nativa para ARM64 via NDK / C++.
  - Tratamento de ciclo de vida móvel: suspensão em segundo plano, controle térmico (thermal throttling) e prevenção de encerramento por Out-Of-Memory (OOM Killer do Android).

---

## 3. Esclarecimento Rigoroso das Métricas e Inconsistências

Para evitar que metas futuras de compatibilidade sejam interpretadas como resultados já homologados:

### Inconsistência 1: Compatibilidade 4–8 GB de RAM vs Hardware Avaliado
* **O que foi medido e comprovado**: Os relatórios publicados e tabelas de velocidade (ex: 61,1 tok/s e 9,5 tok/s) foram mensurados em uma GPU **NVIDIA GeForce RTX 3060 12GB**.
* **O que é o target CPU**: A execução do modelo MoE de 35B consumindo apenas 3 a 4 GB de RAM em CPU de entrada é a **meta de engenharia** do marco **LEGION CPU-First**, que depende da eliminação da materialização das 328 matrizes densas.

### Inconsistência 2: Identificação Exata dos Artefatos MoE (Tamanho & SHA-256)
A tabela a seguir consolida a auditoria formal dos artefatos para evitar ambiguidade entre GiB binário (1024³) e GB decimal (1000³):

| Artefato | Arquivo | Tamanho Exato em Bytes | Tamanho em GiB (1024³) | Tamanho em GB (1000³) | SHA-256 | Configuração e Finalidade |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Artefato A** | `qwen35b_full.vq2m` | `10.102.102.496` bytes | 9,408 GiB | 10,10 GB | Verificado | Baseline 100% 2.0 bpw em todos os 10.240 experts. |
| **Artefato B1** | `qwen35b_mixed_b.vq2m` | `10.104.461.792` bytes | 9,411 GiB | 10,10 GB | Verificado | Iteração intermediária (10 experts promovidos). |
| **🏆 Artefato B2** | `qwen35b_mixed_b2.vq2m` | `10.108.393.952` bytes | 9,414 GiB | 10,11 GB | `2ADB36E37F387842A521D2F3E07A29B77A7D9240997411372E0975586AEB8EAE` | **Candidato Produção Oficial**: 16 experts âncora protegidos (L31–L39), KL = 0,0176 (-75,4%). |

---

## 4. O Plano de Ação Imediato: Próximo Marco Central

Não tentaremos desenvolver CUDA, CPU e Android simultaneamente. A prioridade é resolver o problema que conseguimos medir hoje no hardware do desenvolvedor:

### Marco: LEGION CPU-First
* **Objetivo Geral**: Executar o artefato B2 congelado sem manter as 328 projeções densas reconstruídas na RAM.
* **Meta 1**: Concluir o benchmark CPU atual do baseline e registrar os valores reais de tempo e RSS.
* **Meta 2**: Localizar a origem exata das 328 matrizes BF16 no código de carga e substituir por execução de baixa pegada.
* **Meta 3**: Medir o ganho de memória e velocidade obtido antes de introduzir instruções SIMD customizadas.
* **Meta 4**: Preservar a integridade do backend CUDA e verificar que o hash SHA-256 do contêiner B2 permanece absolutamente inalterado.

> **Princípio Fundamental**: Não precisamos recomprimir o modelo nem reduzir a qualidade do B2. O peso comprimido já é ótimo — precisamos mudar a forma como o runtime CPU executa esses pesos.
