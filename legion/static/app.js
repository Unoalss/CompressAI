// legion/static/app.js - Lógica interativa do Chatbot LEGION

const messagesArea = document.getElementById('messagesArea');
const chatForm = document.getElementById('chatForm');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearBtn');

const driveLabel = document.getElementById('driveLabel');
const totalExperts = document.getElementById('totalExperts');
const activePerToken = document.getElementById('activePerToken');
const distinctExperts = document.getElementById('distinctExperts');
const cachePoolText = document.getElementById('cachePoolText');
const hitRateText = document.getElementById('hitRateText');
const hitRateFill = document.getElementById('hitRateFill');
const vramText = document.getElementById('vramText');
const speedText = document.getElementById('speedText');
const stepMsText = document.getElementById('stepMsText');
const expertsMatrix = document.getElementById('expertsMatrix');
const pulseDot = document.getElementById('pulseDot');

let isGenerating = false;

// Inicializa a grade do Radar MoE (128 pequenos indicadores)
function initRadar() {
    expertsMatrix.innerHTML = '';
    for (let i = 0; i < 128; i++) {
        const dot = document.createElement('div');
        dot.className = 'expert-dot';
        dot.id = `expert-dot-${i}`;
        expertsMatrix.appendChild(dot);
    }
}
initRadar();

function animateRadarActivity(hitRate) {
    const dots = document.querySelectorAll('.expert-dot');
    dots.forEach(d => {
        if (Math.random() < 0.35) {
            d.className = 'expert-dot ' + (Math.random() < (hitRate / 100) ? 'hit' : 'miss');
        } else {
            d.className = 'expert-dot';
        }
    });
}

// Atualiza status do hardware
async function updateStatus() {
    try {
        const res = await fetch('/api/status');
        if (!res.ok) return;
        const data = await res.json();
        
        if (data.source) driveLabel.textContent = data.source;
        if (data.total_experts) totalExperts.textContent = Number(data.total_experts).toLocaleString('pt-BR');
        if (data.active_per_token && activePerToken) {
            activePerToken.textContent = `${data.active_per_token} especialistas (Top-8 Nativo)`;
        }
        if (data.distinct_activated && distinctExperts) {
            distinctExperts.textContent = `${data.distinct_activated} de 10.240`;
        }
        if (data.cache_total_vram && cachePoolText) {
            cachePoolText.textContent = `${Number(data.cache_total_vram).toLocaleString('pt-BR')} experts (Zero-Copy)`;
        }
        if (data.hit_rate_pct !== undefined) {
            hitRateText.textContent = `${data.hit_rate_pct}%`;
            hitRateFill.style.width = `${data.hit_rate_pct}%`;
        }
        if (data.vram_mb) {
            vramText.textContent = `${(data.vram_mb / 1024).toFixed(1)} GB`;
        }
    } catch (e) {
        // Silencioso se offline
    }
}
setInterval(updateStatus, 3000);
updateStatus();

// Input Auto-resize
userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 140) + 'px';
});

userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});

function usePrompt(text) {
    userInput.value = text;
    userInput.focus();
    chatForm.dispatchEvent(new Event('submit'));
}

clearBtn.addEventListener('click', () => {
    messagesArea.innerHTML = `
        <div class="message assistant-message">
            <div class="avatar">L</div>
            <div class="message-content">
                <p><strong>Conversa reiniciada.</strong></p>
                <p>A Legião de 10.240 especialistas está pronta no pendrive para sua próxima pergunta.</p>
            </div>
        </div>
    `;
});

function appendMessage(role, text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}-message`;
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = role === 'user' ? 'U' : 'L';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = `<p>${escapeHtml(text)}</p>`;
    
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(content);
    messagesArea.appendChild(msgDiv);
    messagesArea.scrollTop = messagesArea.scrollHeight;
    
    return content;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/\n/g, '<br>');
}

// Submissão do Chat via SSE
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = userInput.value.trim();
    if (!text || isGenerating) return;

    appendMessage('user', text);
    userInput.value = '';
    userInput.style.height = 'auto';

    isGenerating = true;
    sendBtn.disabled = true;
    pulseDot.style.backgroundColor = 'var(--neon-cyan)';

    const assistantContent = appendMessage('assistant', '');
    let accumulatedText = '';
    let lastStats = null;

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, max_tokens: 300 })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        let isDone = false;

        while (!isDone) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const raw = line.slice(6).trim();
                    if (raw === '[DONE]') {
                        isDone = true;
                        break;
                    }

                    try {
                        const payload = JSON.parse(raw);
                        if (payload.token) {
                            accumulatedText += payload.token;
                            assistantContent.innerHTML = `<p>${escapeHtml(accumulatedText)}</p>`;
                            messagesArea.scrollTop = messagesArea.scrollHeight;
                        }
                        if (payload.stats) {
                            lastStats = payload.stats;
                            if (lastStats.tok_s && speedText) speedText.textContent = `${lastStats.tok_s} tok/s`;
                            if (lastStats.step_ms && stepMsText) stepMsText.textContent = `${lastStats.step_ms} ms`;
                            if (lastStats.distinct_activated && distinctExperts) {
                                distinctExperts.textContent = `${lastStats.distinct_activated} de 10.240`;
                            }
                            if (lastStats.hit_rate_pct !== undefined) {
                                hitRateText.textContent = `${lastStats.hit_rate_pct}%`;
                                hitRateFill.style.width = `${lastStats.hit_rate_pct}%`;
                                animateRadarActivity(lastStats.hit_rate_pct);
                            }
                            if (lastStats.vram_mb) {
                                vramText.textContent = `${(lastStats.vram_mb / 1024).toFixed(1)} GB`;
                            }
                        }
                    } catch (err) {
                        // Ignora parsing chunks intermediários
                    }
                }
            }
        }
        try { await reader.cancel(); } catch (e) {}

        // Adiciona chip de telemetria final
        if (lastStats) {
            const chip = document.createElement('div');
            chip.className = 'telemetry-chip';
            chip.innerHTML = `
                <span>Fonte: <strong>${lastStats.source || 'Pendrive D:\\'}</strong></span>
                <span>Modo Neural: <strong>${lastStats.active_per_token || 320} ativos/token (${lastStats.distinct_activated || 320} únicos)</strong></span>
                <span>Velocidade: <strong>${lastStats.tok_s || '0.0'} tok/s</strong></span>
                <span>Latência: <strong>${lastStats.step_ms || '0'} ms/token</strong></span>
                <span>VRAM: <strong>${(lastStats.vram_mb / 1024).toFixed(1)} GB (-85%)</strong></span>
                <span>Cache Hit: <strong>${lastStats.hit_rate_pct}%</strong></span>
            `;
            assistantContent.appendChild(chip);
        }

    } catch (err) {
        assistantContent.innerHTML += `<p style="color: #ef4444;">Erro na conexão com a legião de especialistas: ${err.message}</p>`;
    } finally {
        isGenerating = false;
        sendBtn.disabled = false;
        pulseDot.style.backgroundColor = 'var(--neon-green)';
        userInput.focus();
    }
});
