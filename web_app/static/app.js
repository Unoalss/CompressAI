/**
 * VQ2 Quantum Studio - Client-Side App Logic
 * Real-time SSE token streaming, MoE 256-Expert Grid animation & hardware telemetry.
 */

document.addEventListener("DOMContentLoaded", () => {
    // Elementos DOM
    const chatMessages = document.getElementById("chat-messages");
    const chatForm = document.getElementById("chat-form");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const expertGrid = document.getElementById("expert-grid");
    const liveSpeed = document.getElementById("live-speed");
    const layerLatency = document.getElementById("layer-latency");
    const tempSlider = document.getElementById("temp-slider");
    const tempVal = document.getElementById("temp-val");
    const maxTokSlider = document.getElementById("max-tok-slider");
    const maxTokVal = document.getElementById("max-tok-val");
    const mtpToggle = document.getElementById("mtp-toggle");
    const presetChips = document.querySelectorAll(".preset-chip");

    let isGenerating = false;
    let currentReader = null;

    // 1. Inicializa a Grade MoE de 256 Experts (16x16)
    const expertNodes = [];
    for (let i = 0; i < 256; i++) {
        const node = document.createElement("div");
        node.className = "expert-node";
        node.dataset.id = i;
        node.title = `Expert #${i}`;
        expertGrid.appendChild(node);
        expertNodes.push(node);
    }

    // 2. Sliders e Eventos de Input
    if (tempSlider) {
        tempSlider.addEventListener("input", (e) => {
            tempVal.textContent = e.target.value;
        });
    }

    if (maxTokSlider) {
        maxTokSlider.addEventListener("input", (e) => {
            maxTokVal.textContent = e.target.value;
        });
    }

    // Auto-ajuste de altura da textarea
    userInput.addEventListener("input", () => {
        userInput.style.height = "auto";
        userInput.style.height = Math.min(userInput.scrollHeight, 160) + "px";
    });

    // Enter envia a mensagem, Shift+Enter pula linha
    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Submissão do formulário
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        sendMessage();
    });

    // 3. Preset Chips
    presetChips.forEach(chip => {
        chip.addEventListener("click", () => {
            if (isGenerating) return;
            const prompt = chip.getAttribute("data-prompt");
            userInput.value = prompt;
            sendMessage();
        });
    });

    // 4. Carrega Status Inicial do Backend
    async function fetchStatus() {
        try {
            const res = await fetch("/api/status");
            if (res.ok) {
                const data = await res.json();
                const gpuEl = document.getElementById("gpu-name");
                const vFreeEl = document.getElementById("vram-free");
                const vTextEl = document.getElementById("vram-text");
                if (gpuEl) gpuEl.textContent = data.gpu || "RTX 3060 12GB";
                if (vFreeEl) vFreeEl.textContent = `${data.vram_free_gb} GB`;
                if (vTextEl) vTextEl.textContent = `${data.vram_model_gb} / ${data.vram_total_gb} GB`;
                if (liveSpeed) liveSpeed.textContent = data.mtp_tok_s ? data.mtp_tok_s.toFixed(1) : "61.1";
            }
        } catch (e) {
            console.warn("Backend local offline, usando valores padrão de hardware.", e);
        }
    }
    fetchStatus();

    // 5. Função de Animação dos Experts Ativos no Grid
    function highlightActiveExperts(activeIds) {
        expertNodes.forEach(n => n.classList.remove("active"));
        if (Array.isArray(activeIds)) {
            activeIds.forEach(id => {
                if (expertNodes[id]) {
                    expertNodes[id].classList.add("active");
                }
            });
        }
    }

    // 6. Função Principal de Envio e Desbloqueio Garantido
    async function sendMessage() {
        if (isGenerating) return;
        const promptText = userInput.value.trim();
        if (!promptText) return;

        isGenerating = true;

        // Limpa campo e desabilita temporariamente enquanto gera
        userInput.value = "";
        userInput.style.height = "auto";
        userInput.disabled = true;
        sendBtn.disabled = true;

        // Adiciona Bolha do Usuário
        appendUserMessage(promptText);

        // Prepara Bolha de Resposta do Assistente
        const assistantMsgEl = createAssistantMessage();
        chatMessages.appendChild(assistantMsgEl);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        const textContainer = assistantMsgEl.querySelector(".message-text");
        const metaContainer = assistantMsgEl.querySelector(".message-meta");

        const requestPayload = {
            prompt: promptText,
            max_tokens: maxTokSlider ? parseInt(maxTokSlider.value) : 256,
            temperature: tempSlider ? parseFloat(tempSlider.value) : 0.7,
            use_mtp: mtpToggle ? mtpToggle.checked : true
        };

        // Fallback de segurança: desbloqueia a caixa caso o stream demore mais de 180 segundos
        const safetyTimeout = setTimeout(() => {
            unlockInput();
        }, 180000);

        try {
            const response = await fetch("/api/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(requestPayload)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            currentReader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";
            let fullText = "";
            let streamFinished = false;

            while (!streamFinished) {
                const { done, value } = await currentReader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                buffer = lines.pop();

                for (const block of lines) {
                    if (block.startsWith("data: ")) {
                        const jsonStr = block.replace("data: ", "").trim();
                        if (!jsonStr) continue;

                        try {
                            const data = JSON.parse(jsonStr);

                            if (data.type === "token") {
                                fullText += data.token;
                                textContainer.innerHTML = renderMarkdown(fullText);

                                if (data.tok_s && liveSpeed) {
                                    liveSpeed.textContent = data.tok_s.toFixed(1);
                                }
                                if (data.layer_ms && layerLatency) {
                                    layerLatency.textContent = `${data.layer_ms} ms`;
                                }
                                if (data.active_experts) {
                                    highlightActiveExperts(data.active_experts);
                                }
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                            } else if (data.type === "done") {
                                streamFinished = true;
                                highlightActiveExperts([]);
                                metaContainer.innerHTML = `
                                    <span class="emerald">⚡ ${data.final_tok_s || 6.3} tok/s (Real)</span> • 
                                    <span>${data.total_tokens} tokens em ${data.elapsed_s}s</span> • 
                                    <span class="cyan">RTX 3060 12GB (35.5B In-Core)</span>
                                `;
                                break;
                            }
                        } catch (err) {
                            console.error("Erro no parsing do payload SSE:", err);
                        }
                    }
                }
            }

            if (currentReader) {
                try { await currentReader.cancel(); } catch (e) {}
            }
        } catch (error) {
            textContainer.innerHTML = `<span style="color:#ff6b6b;">Erro na geração: ${escapeHtml(error.message)}</span>`;
        } finally {
            clearTimeout(safetyTimeout);
            unlockInput();
        }
    }

    function unlockInput() {
        isGenerating = false;
        currentReader = null;
        userInput.disabled = false;
        sendBtn.disabled = false;
        highlightActiveExperts([]);
        userInput.focus();
    }

    // 7. Helpers de Interface e Renderização Markdown
    function appendUserMessage(text) {
        const msg = document.createElement("div");
        msg.className = "message user-message";
        msg.innerHTML = `
            <div class="avatar user-avatar">
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                    <circle cx="12" cy="7" r="4"></circle>
                </svg>
            </div>
            <div class="message-body">
                <div class="message-author">Você</div>
                <div class="message-text">${escapeHtml(text)}</div>
            </div>
        `;
        chatMessages.appendChild(msg);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function createAssistantMessage() {
        const msg = document.createElement("div");
        msg.className = "message assistant-message";
        msg.innerHTML = `
            <div class="avatar assistant-avatar">
                <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                    <polyline points="2 17 12 22 22 17"></polyline>
                    <polyline points="2 12 12 17 22 12"></polyline>
                </svg>
            </div>
            <div class="message-body">
                <div class="message-author">Qwen3.8-35B (VQ2 RDO)</div>
                <div class="message-text"><span class="thinking-indicator"><span class="pulse-dot"></span> Processando nas 40 camadas da RTX 3060...</span></div>
                <div class="message-meta">Acelerando 35.5B parâmetros na GPU...</div>
            </div>
        `;
        return msg;
    }

    // Renderizador de Markdown simples e robusto com formatação de código
    function renderMarkdown(md) {
        if (!md) return "";

        // Bloco de código ```linguagem ... ```
        let html = md.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
            const safeCode = escapeHtml(code.trim());
            const langLabel = lang ? lang.toUpperCase() : "CÓDIGO";
            return `
                <div class="code-block-wrapper" style="margin: 12px 0; border: 1px solid rgba(255,255,255,0.15); border-radius: 8px; background: #080a10; overflow: hidden;">
                    <div style="background: rgba(255,255,255,0.06); padding: 6px 12px; font-size: 0.72rem; font-weight: 700; color: #00f2fe; display: flex; justify-content: space-between; align-items: center;">
                        <span>${langLabel}</span>
                        <button onclick="navigator.clipboard.writeText(this.closest('.code-block-wrapper').querySelector('code').innerText); this.textContent='Copiado!'; setTimeout(()=>this.textContent='Copiar', 2000);" style="background: rgba(0,242,254,0.15); border: 1px solid rgba(0,242,254,0.3); color: #00f2fe; padding: 2px 8px; border-radius: 4px; cursor: pointer; font-size: 0.7rem;">Copiar</button>
                    </div>
                    <pre style="margin: 0; padding: 12px; overflow-x: auto;"><code style="font-family: Consolas, monospace; font-size: 0.85rem; color: #e2e8f0;">${safeCode}</code></pre>
                </div>
            `;
        });

        // Código inline `codigo`
        html = html.replace(/`([^`]+)`/g, '<code style="background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px; color: #00f2fe; font-family: monospace;">$1</code>');

        // Negrito **texto**
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Itálico *texto*
        html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

        // Bloco de pensamento e raciocínio <think>...</think>
        html = html.replace(/<think>([\s\S]*?)<\/think>/gi, (match, thought) => {
            return `
                <details class="thought-bubble" open style="margin: 10px 0; padding: 10px 14px; background: rgba(0, 242, 254, 0.05); border-left: 3px solid #00f2fe; border-radius: 8px; font-size: 0.84rem; color: #94a3b8;">
                    <summary style="cursor: pointer; font-weight: 600; color: #00f2fe; margin-bottom: 6px; user-select: none;">🧠 Raciocínio Interno (Deep Thinking)</summary>
                    <div style="font-style: italic; white-space: pre-wrap; line-height: 1.5;">${escapeHtml(thought.trim())}</div>
                </details>
            `;
        });
        if (html.includes("<think>")) {
            html = html.replace(/<think>([\s\S]*)/gi, (match, thought) => {
                return `
                    <div class="thought-bubble" style="margin: 10px 0; padding: 10px 14px; background: rgba(0, 242, 254, 0.05); border-left: 3px solid #00f2fe; border-radius: 8px; font-size: 0.84rem; color: #94a3b8;">
                        <div style="font-weight: 600; color: #00f2fe; margin-bottom: 6px;">🧠 Raciocínio em andamento...</div>
                        <div style="font-style: italic; white-space: pre-wrap; line-height: 1.5;">${escapeHtml(thought.trim())}</div>
                    </div>
                `;
            });
        }

        // Quebras de linha (preservando parágrafos fora dos blocos de código)
        html = html.replace(/\n\n/g, '<br><br>');
        html = html.replace(/\n/g, '<br>');

        return html;
    }

    function escapeHtml(string) {
        return String(string)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
