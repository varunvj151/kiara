/**
 * Kiara Autonomous Foundation Model — Web Application Logic
 * =========================================================================
 * Full client-side state machine:
 * - SSE Streaming Consumer (fetch + ReadableStream)
 * - Real-time Token-by-Token Rendering with Blinking Cursor
 * - Markdown & Code Syntax Formatter with One-Click Copy
 * - Conversation Session Persistence (localStorage)
 * - Settings Synchronization & Dynamic Checkpoint Switcher
 * - Keyboard Shortcuts & Responsive Drawer Navigation
 */

(() => {
  'use strict';

  // --- State Management ---
  const state = {
    sessions: [],
    currentSessionId: null,
    isGenerating: false,
    abortController: null,
    modelStatus: null,
    settings: {
      temperature: 0.7,
      maxTokens: 100,
      topK: 40,
      instruct: true,
      activeCheckpoint: '',
    },
  };

  // --- DOM Elements ---
  const el = {
    // Layout & Navigation
    sidebar: document.getElementById('sidebar'),
    mobileMenuBtn: document.getElementById('mobileMenuBtn'),
    closeSidebarBtn: document.getElementById('closeSidebarBtn'),
    newChatBtn: document.getElementById('newChatBtn'),
    chatHistoryList: document.getElementById('chatHistoryList'),
    chatTitle: document.getElementById('chatTitle'),
    activeLabel: document.getElementById('activeLabel'),
    statusText: document.getElementById('statusText'),
    modelParamText: document.getElementById('modelParamText'),
    modelDeviceText: document.getElementById('modelDeviceText'),

    // Chat Arena
    chatViewport: document.getElementById('chatViewport'),
    heroWelcome: document.getElementById('heroWelcome'),
    messagesList: document.getElementById('messagesList'),
    starterCards: document.querySelectorAll('.starter-card'),

    // Input Bar
    chatForm: document.getElementById('chatForm'),
    promptInput: document.getElementById('promptInput'),
    sendBtn: document.getElementById('sendBtn'),
    stopBtn: document.getElementById('stopBtn'),
    tempPill: document.getElementById('tempPill'),
    tokensPill: document.getElementById('tokensPill'),
    instructPill: document.getElementById('instructPill'),

    // Settings Drawer
    settingsTriggerBtn: document.getElementById('settingsTriggerBtn'),
    topSettingsBtn: document.getElementById('topSettingsBtn'),
    closeDrawerBtn: document.getElementById('closeDrawerBtn'),
    drawerBackdrop: document.getElementById('drawerBackdrop'),
    settingsDrawer: document.getElementById('settingsDrawer'),
    checkpointSelect: document.getElementById('checkpointSelect'),
    tempSlider: document.getElementById('tempSlider'),
    tempVal: document.getElementById('tempVal'),
    tokensSlider: document.getElementById('tokensSlider'),
    tokensVal: document.getElementById('tokensVal'),
    topkSlider: document.getElementById('topkSlider'),
    topkVal: document.getElementById('topkVal'),
    instructToggle: document.getElementById('instructToggle'),
    clearChatBtn: document.getElementById('clearChatBtn'),
  };

  // --- Storage Helpers ---
  const STORAGE_KEYS = {
    SESSIONS: 'kiara_sessions_v1',
    CURRENT_ID: 'kiara_current_session_id',
    SETTINGS: 'kiara_settings_v1',
  };

  function loadSavedState() {
    try {
      const savedSessions = localStorage.getItem(STORAGE_KEYS.SESSIONS);
      if (savedSessions) {
        state.sessions = JSON.parse(savedSessions);
      }
      const savedCurrentId = localStorage.getItem(STORAGE_KEYS.CURRENT_ID);
      if (savedCurrentId) {
        state.currentSessionId = savedCurrentId;
      }
      const savedSettings = localStorage.getItem(STORAGE_KEYS.SETTINGS);
      if (savedSettings) {
        state.settings = { ...state.settings, ...JSON.parse(savedSettings) };
      }
    } catch (e) {
      console.warn('Failed to parse localStorage:', e);
    }

    // Ensure at least one session exists
    if (!state.sessions.length) {
      createNewSession();
    } else if (!state.currentSessionId || !state.sessions.find(s => s.id === state.currentSessionId)) {
      state.currentSessionId = state.sessions[0].id;
    }
  }

  function persistState() {
    try {
      localStorage.setItem(STORAGE_KEYS.SESSIONS, JSON.stringify(state.sessions));
      localStorage.setItem(STORAGE_KEYS.CURRENT_ID, state.currentSessionId);
      localStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(state.settings));
    } catch (e) {
      console.warn('Failed to persist localStorage:', e);
    }
  }

  function getCurrentSession() {
    return state.sessions.find(s => s.id === state.currentSessionId);
  }

  function createNewSession() {
    const newSession = {
      id: 'session_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7),
      title: 'New Conversation',
      createdAt: new Date().toISOString(),
      messages: [],
    };
    state.sessions.unshift(newSession);
    state.currentSessionId = newSession.id;
    persistState();
    renderSidebarHistory();
    renderCurrentConversation();
    el.promptInput.focus();
    if (window.innerWidth <= 900) {
      el.sidebar.classList.remove('mobile-open');
    }
  }

  function deleteSession(sessionId, event) {
    if (event) event.stopPropagation();
    state.sessions = state.sessions.filter(s => s.id !== sessionId);
    if (!state.sessions.length) {
      createNewSession();
    } else if (state.currentSessionId === sessionId) {
      state.currentSessionId = state.sessions[0].id;
    }
    persistState();
    renderSidebarHistory();
    renderCurrentConversation();
  }

  function switchSession(sessionId) {
    if (state.isGenerating) return;
    state.currentSessionId = sessionId;
    persistState();
    renderSidebarHistory();
    renderCurrentConversation();
    if (window.innerWidth <= 900) {
      el.sidebar.classList.remove('mobile-open');
    }
  }

  // --- Markdown Formatter ---
  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function renderMarkdown(rawText) {
    if (!rawText) return '';

    // Split code blocks first
    const codeBlocks = [];
    let text = rawText.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
      const id = '__CODE_BLOCK_' + codeBlocks.length + '__';
      codeBlocks.push({ lang: lang.trim() || 'text', code: code });
      return id;
    });

    // Escape remaining HTML to prevent XSS
    text = escapeHtml(text);

    // Format bold: **text**
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Format italic: *text*
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Format inline code: `code`
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Format paragraphs & line breaks
    const lines = text.split('\n');
    let inList = false;
    let formattedHtml = '';

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();

      // Check bullet items
      if (line.startsWith('- ') || line.startsWith('* ')) {
        if (!inList) {
          formattedHtml += '<ul>';
          inList = true;
        }
        formattedHtml += `<li>${line.substring(2)}</li>`;
      } else {
        if (inList) {
          formattedHtml += '</ul>';
          inList = false;
        }
        if (line.length > 0) {
          formattedHtml += `<p>${line}</p>`;
        }
      }
    }
    if (inList) formattedHtml += '</ul>';

    // Reinsert code blocks
    for (let i = 0; i < codeBlocks.length; i++) {
      const { lang, code } = codeBlocks[i];
      const escapedCode = escapeHtml(code);
      const codeBlockMarkup = `
        <div class="code-block-container">
          <div class="code-block-header">
            <span>${lang}</span>
            <button class="copy-code-btn" type="button" onclick="window.kiaraCopyCode(this)">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
              <span>Copy</span>
            </button>
          </div>
          <pre class="code-block-pre"><code>${escapedCode}</code></pre>
        </div>
      `;
      formattedHtml = formattedHtml.replace('__CODE_BLOCK_' + i + '__', codeBlockMarkup);
    }

    return formattedHtml;
  }

  // Global helper for code block copy buttons
  window.kiaraCopyCode = (btn) => {
    const pre = btn.closest('.code-block-container').querySelector('code');
    if (!pre) return;
    const text = pre.innerText;
    navigator.clipboard.writeText(text).then(() => {
      const span = btn.querySelector('span');
      const original = span.textContent;
      span.textContent = 'Copied!';
      btn.style.color = '#10b981';
      setTimeout(() => {
        span.textContent = original;
        btn.style.color = '';
      }, 2000);
    });
  };

  // --- Rendering UI ---
  function renderSidebarHistory() {
    el.chatHistoryList.innerHTML = '';
    state.sessions.forEach(session => {
      const item = document.createElement('div');
      item.className = 'history-item' + (session.id === state.currentSessionId ? ' active' : '');
      item.innerHTML = `
        <span class="history-item-title" title="${escapeHtml(session.title)}">${escapeHtml(session.title)}</span>
        <button class="history-delete-btn" title="Delete chat" aria-label="Delete">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
        </button>
      `;
      item.addEventListener('click', () => switchSession(session.id));
      const delBtn = item.querySelector('.history-delete-btn');
      delBtn.addEventListener('click', (e) => deleteSession(session.id, e));
      el.chatHistoryList.appendChild(item);
    });
  }

  function renderCurrentConversation() {
    const session = getCurrentSession();
    if (!session) return;

    el.chatTitle.textContent = session.title;

    if (!session.messages || session.messages.length === 0) {
      el.heroWelcome.style.display = 'flex';
      el.messagesList.innerHTML = '';
    } else {
      el.heroWelcome.style.display = 'none';
      el.messagesList.innerHTML = '';
      session.messages.forEach(msg => {
        appendMessageElement(msg, false);
      });
      scrollToBottom();
    }
  }

  function appendMessageElement(msg, isLive = false) {
    const isUser = msg.role === 'user';
    const messageNode = document.createElement('div');
    messageNode.className = `message ${isUser ? 'user-message' : 'assistant-message'}`;
    if (msg.id) messageNode.id = msg.id;

    const avatarSvg = isUser
      ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`
      : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm1 14.93V18a1 1 0 0 1-2 0v-1.07A7 7 0 0 1 5.07 11H4a1 1 0 0 1 0-2h1.07A7 7 0 0 1 11 3.07V2a1 1 0 0 1 2 0v1.07A7 7 0 0 1 18.93 9H20a1 1 0 0 1 0 2h-1.07A7 7 0 0 1 13 16.93z"/></svg>`;

    const timeFormatted = msg.timestamp
      ? new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      : '';

    let contentHtml = isUser ? escapeHtml(msg.content) : renderMarkdown(msg.content);
    if (isLive) {
      contentHtml += '<span class="cursor-blink"></span>';
    }

    let metaFooterHtml = '';
    if (!isUser && msg.metrics) {
      metaFooterHtml = `
        <div class="message-meta-footer">
          <span class="meta-pill">⚡ ${msg.metrics.tokPerSec || 0} tok/s</span>
          <span class="meta-pill">🔢 ${msg.metrics.tokens || 0} tokens</span>
          <span class="meta-pill">⏱️ ${msg.metrics.elapsed || 0}s</span>
        </div>
      `;
    }

    messageNode.innerHTML = `
      <div class="message-avatar">
        ${avatarSvg}
      </div>
      <div class="message-content-wrapper">
        <div class="message-header">
          <span class="message-author">${isUser ? 'You' : 'Kiara'}</span>
          <span class="message-time">${timeFormatted}</span>
        </div>
        <div class="message-bubble" id="${msg.id ? msg.id + '-bubble' : ''}">
          ${contentHtml}
        </div>
        ${metaFooterHtml}
      </div>
    `;

    el.messagesList.appendChild(messageNode);
    return messageNode;
  }

  function scrollToBottom() {
    el.chatViewport.scrollTop = el.chatViewport.scrollHeight;
  }

  // --- Chat Submission & SSE Streaming ---
  async function handleChatSubmit(e) {
    if (e) e.preventDefault();
    if (state.isGenerating) return;

    const rawPrompt = el.promptInput.value.trim();
    if (!rawPrompt) return;

    const session = getCurrentSession();
    if (!session) return;

    // Set first message title
    if (!session.messages || session.messages.length === 0) {
      session.title = rawPrompt.slice(0, 36) + (rawPrompt.length > 36 ? '...' : '');
      renderSidebarHistory();
      el.chatTitle.textContent = session.title;
      el.heroWelcome.style.display = 'none';
    }

    // Append User Message
    const userMsg = {
      id: 'msg_' + Date.now(),
      role: 'user',
      content: rawPrompt,
      timestamp: new Date().toISOString(),
    };
    session.messages.push(userMsg);
    appendMessageElement(userMsg);
    persistState();

    // Reset input
    el.promptInput.value = '';
    resizeTextarea();
    scrollToBottom();

    // Prepare Assistant Message Placeholder
    const assistantMsgId = 'msg_ast_' + Date.now();
    const assistantMsg = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
      metrics: null,
    };
    session.messages.push(assistantMsg);
    const astNode = appendMessageElement(assistantMsg, true);
    const bubbleEl = document.getElementById(assistantMsgId + '-bubble');

    // UI state for generation
    setGeneratingState(true);

    state.abortController = new AbortController();

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: rawPrompt,
          max_tokens: state.settings.maxTokens,
          temperature: state.settings.temperature,
          top_k: state.settings.topK,
          instruct: state.settings.instruct,
        }),
        signal: state.abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let partialBuffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        partialBuffer += decoder.decode(value, { stream: true });
        const lines = partialBuffer.split('\n');
        partialBuffer = lines.pop(); // Keep unfinished line in buffer

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;

          const jsonStr = trimmed.substring(5).trim();
          if (!jsonStr) continue;

          try {
            const data = JSON.parse(jsonStr);

            if (data.token) {
              assistantMsg.content += data.token;
              bubbleEl.innerHTML = renderMarkdown(assistantMsg.content) + '<span class="cursor-blink"></span>';
              scrollToBottom();
            }

            if (data.done) {
              assistantMsg.metrics = {
                tokens: data.tokens_generated,
                elapsed: data.elapsed_sec,
                tokPerSec: data.tok_per_sec,
              };
            }
          } catch (err) {
            console.error('Error parsing SSE data chunk:', err);
          }
        }
      }

      // Finalize message rendering without cursor
      bubbleEl.innerHTML = renderMarkdown(assistantMsg.content);

      if (assistantMsg.metrics) {
        const wrapper = astNode.querySelector('.message-content-wrapper');
        const footer = document.createElement('div');
        footer.className = 'message-meta-footer';
        footer.innerHTML = `
          <span class="meta-pill">⚡ ${assistantMsg.metrics.tokPerSec || 0} tok/s</span>
          <span class="meta-pill">🔢 ${assistantMsg.metrics.tokens || 0} tokens</span>
          <span class="meta-pill">⏱️ ${assistantMsg.metrics.elapsed || 0}s</span>
        `;
        wrapper.appendChild(footer);
      }

    } catch (err) {
      if (err.name === 'AbortError') {
        bubbleEl.innerHTML = renderMarkdown(assistantMsg.content) + '<p><em>[Generation stopped by user]</em></p>';
      } else {
        bubbleEl.innerHTML = `<p style="color: var(--status-error);">⚠️ Error: ${escapeHtml(err.message)}</p>`;
      }
    } finally {
      persistState();
      setGeneratingState(false);
      scrollToBottom();
    }
  }

  function setGeneratingState(generating) {
    state.isGenerating = generating;
    if (generating) {
      el.sendBtn.style.display = 'none';
      el.stopBtn.style.display = 'flex';
      el.promptInput.disabled = true;
    } else {
      el.sendBtn.style.display = 'flex';
      el.stopBtn.style.display = 'none';
      el.promptInput.disabled = false;
      el.promptInput.focus();
    }
  }

  function handleStopGeneration() {
    if (state.abortController) {
      state.abortController.abort();
    }
  }

  // --- Auto-resize Textarea ---
  function resizeTextarea() {
    el.promptInput.style.height = 'auto';
    const newHeight = Math.min(el.promptInput.scrollHeight, 200);
    el.promptInput.style.height = Math.max(28, newHeight) + 'px';
  }

  // --- Settings & Checkpoint Management ---
  async function fetchServerStatus() {
    try {
      const res = await fetch('/api/status');
      if (!res.ok) throw new Error('Status failed');
      const data = await res.json();
      state.modelStatus = data;

      // Update UI elements
      el.statusText.textContent = 'Model Online';
      el.statusText.style.color = 'var(--status-online)';
      el.modelParamText.textContent = `${(data.parameters / 1e6).toFixed(1)}M Params`;
      el.modelDeviceText.textContent = data.device.toUpperCase();
      el.activeLabel.textContent = data.checkpoint || 'Kiara';

      // Populate Checkpoints
      el.checkpointSelect.innerHTML = '';
      if (data.available_checkpoints && data.available_checkpoints.length) {
        const currentNorm = (data.checkpoint_path || '').replace(/\\/g, '/').toLowerCase();
        data.available_checkpoints.forEach(ckpt => {
          const opt = document.createElement('option');
          opt.value = ckpt.path;
          opt.textContent = `${ckpt.name} (${ckpt.size_mb} MB)`;
          const ckptNorm = (ckpt.path || '').replace(/\\/g, '/').toLowerCase();
          if (currentNorm && currentNorm === ckptNorm) {
            opt.selected = true;
          }
          el.checkpointSelect.appendChild(opt);
        });
      } else {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = 'Default Checkpoint';
        el.checkpointSelect.appendChild(opt);
      }
    } catch (e) {
      el.statusText.textContent = 'Offline (Dev Mode)';
      el.statusText.style.color = 'var(--status-warn)';
      el.modelParamText.textContent = '40.9M';
    }
  }

  async function handleSwitchCheckpoint() {
    const selectedPath = el.checkpointSelect.value;
    if (!selectedPath) return;

    try {
      el.statusText.textContent = 'Switching...';
      const res = await fetch('/api/switch_checkpoint', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: selectedPath }),
      });
      const data = await res.json();
      if (data.success) {
        await fetchServerStatus();
      } else {
        alert('Failed to switch: ' + data.error);
      }
    } catch (e) {
      alert('Error switching checkpoint: ' + e.message);
    }
  }

  function syncSettingsUI() {
    el.tempSlider.value = state.settings.temperature;
    el.tempVal.textContent = state.settings.temperature;
    el.tempPill.textContent = `Temp: ${state.settings.temperature}`;

    el.tokensSlider.value = state.settings.maxTokens;
    el.tokensVal.textContent = state.settings.maxTokens;
    el.tokensPill.textContent = `Max: ${state.settings.maxTokens}`;

    el.topkSlider.value = state.settings.topK;
    el.topkVal.textContent = state.settings.topK;

    el.instructToggle.checked = state.settings.instruct;
    if (state.settings.instruct) {
      el.instructPill.classList.add('active');
    } else {
      el.instructPill.classList.remove('active');
    }
  }

  // --- Event Listeners Setup ---
  function setupEventListeners() {
    // Chat submit & stop
    el.chatForm.addEventListener('submit', handleChatSubmit);
    el.stopBtn.addEventListener('click', handleStopGeneration);

    // Textarea handling
    el.promptInput.addEventListener('input', resizeTextarea);
    el.promptInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleChatSubmit();
      }
    });

    // Global keyboard shortcuts
    window.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        createNewSession();
      }
    });

    // New Chat Button
    el.newChatBtn.addEventListener('click', createNewSession);

    // Starter Cards
    el.starterCards.forEach(card => {
      card.addEventListener('click', () => {
        const prompt = card.getAttribute('data-prompt');
        if (prompt) {
          el.promptInput.value = prompt;
          handleChatSubmit();
        }
      });
    });

    // Drawer Controls
    const openDrawer = () => {
      el.settingsDrawer.classList.add('active');
      el.drawerBackdrop.classList.add('active');
    };
    const closeDrawer = () => {
      el.settingsDrawer.classList.remove('active');
      el.drawerBackdrop.classList.remove('active');
    };

    el.settingsTriggerBtn.addEventListener('click', openDrawer);
    el.topSettingsBtn.addEventListener('click', openDrawer);
    el.closeDrawerBtn.addEventListener('click', closeDrawer);
    el.drawerBackdrop.addEventListener('click', closeDrawer);

    // Mobile Sidebar Controls
    el.mobileMenuBtn.addEventListener('click', () => {
      el.sidebar.classList.toggle('mobile-open');
    });
    el.closeSidebarBtn.addEventListener('click', () => {
      el.sidebar.classList.remove('mobile-open');
    });

    // Slider Listeners
    el.tempSlider.addEventListener('input', (e) => {
      state.settings.temperature = parseFloat(e.target.value);
      el.tempVal.textContent = state.settings.temperature;
      el.tempPill.textContent = `Temp: ${state.settings.temperature}`;
      persistState();
    });

    el.tokensSlider.addEventListener('input', (e) => {
      state.settings.maxTokens = parseInt(e.target.value, 10);
      el.tokensVal.textContent = state.settings.maxTokens;
      el.tokensPill.textContent = `Max: ${state.settings.maxTokens}`;
      persistState();
    });

    el.topkSlider.addEventListener('input', (e) => {
      state.settings.topK = parseInt(e.target.value, 10);
      el.topkVal.textContent = state.settings.topK;
      persistState();
    });

    el.instructToggle.addEventListener('change', (e) => {
      state.settings.instruct = e.target.checked;
      if (state.settings.instruct) {
        el.instructPill.classList.add('active');
      } else {
        el.instructPill.classList.remove('active');
      }
      persistState();
    });

    // Checkpoint Switcher
    el.checkpointSelect.addEventListener('change', handleSwitchCheckpoint);

    // Clear History Button
    el.clearChatBtn.addEventListener('click', () => {
      if (confirm('Clear all conversation history? This cannot be undone.')) {
        state.sessions = [];
        localStorage.removeItem(STORAGE_KEYS.SESSIONS);
        createNewSession();
        closeDrawer();
      }
    });
  }

  // --- Application Bootstrapping ---
  function init() {
    loadSavedState();
    syncSettingsUI();
    renderSidebarHistory();
    renderCurrentConversation();
    setupEventListeners();
    fetchServerStatus();
  }

  // Launch on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
