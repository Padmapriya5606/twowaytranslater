/**
 * ISL Two-Way Translator — Master Application Coordinator
 * Connects UI, Avatar Engine, Camera Pipeline, Speech Recognition/TTS, and Backend APIs
 */

document.addEventListener('DOMContentLoaded', () => {
  // App State
  const state = {
    userRole: 'deaf', // 'deaf', 'hearing', 'dual'
    userName: 'User',
    dialect: 'generic',
    autoTts: true,
    ttsRate: 1.0,
    showSkeleton: true,
    activeTab: 'dual-room',
    dictionaryData: null,
    activeDictWord: 'HELLO'
  };

  // 1. Initialize Engines
  const speechEngine = new SpeechEngine({
    ttsRate: state.ttsRate,
    autoSpeak: state.autoTts,
    onSpeechResult: (text, isFinal) => handleVoiceInput(text, isFinal),
    onListeningChange: (isListening) => updateMicUI(isListening)
  });

  // Dual Room Avatar
  const dualAvatar = new SignAvatarRenderer('dualAvatarCanvas', {
    style: 'humanoid',
    onWordChange: (word) => updateDualAvatarGlossTracker(word),
    onSequenceComplete: () => {
      const stateBadge = document.getElementById('dualAvatarState');
      if (stateBadge) stateBadge.innerHTML = '<i class="fa-solid fa-robot"></i> Avatar Ready';
    }
  });

  // Studio Avatar
  const studioAvatar = new SignAvatarRenderer('studioAvatarCanvas', {
    style: 'humanoid',
    onWordChange: (word, index) => highlightStudioChip(index, word),
    onFrameUpdate: (progress) => {
      const slider = document.getElementById('avatarTimelineSlider');
      if (slider) slider.value = Math.round(progress * 100);
    }
  });

  // Dictionary Preview Avatar
  const dictAvatar = new SignAvatarRenderer('dictAvatarCanvas', {
    style: 'humanoid'
  });

  // Dual Room Camera Pipeline
  const dualCamera = new CameraPipeline('dualVideoElement', 'dualCanvasOverlay', {
    showSkeleton: state.showSkeleton,
    onSignDetected: (pred) => handleDualSignPrediction(pred),
    onFpsUpdate: (fps) => {
      const el = document.getElementById('dualFpsMeter');
      if (el) el.innerText = `${fps} FPS`;
    }
  });

  // Signer Studio Camera Pipeline
  const studioCamera = new CameraPipeline('studioVideo', 'studioCanvas', {
    showSkeleton: state.showSkeleton,
    onSignDetected: (pred) => handleStudioSignPrediction(pred),
    onBufferUpdate: (current, max) => {
      const bar = document.getElementById('bufferProgressBar');
      const text = document.getElementById('bufferCountText');
      if (bar) bar.style.width = `${(current / max) * 100}%`;
      if (text) text.innerText = `${current} / ${max}`;
    },
    onFpsUpdate: (fps) => {
      const el = document.getElementById('studioFps');
      if (el) el.innerText = `${fps} FPS`;
    }
  });

  // ==========================================================================
  // INITIALIZATION & TAB ROUTING
  // ==========================================================================
  initNavigation();
  initOnboarding();
  initSettings();
  initDualRoom();
  initSignerStudio();
  initAvatarStudio();
  initDictionary();
  checkSystemStatus();

  function initNavigation() {
    const tabs = document.querySelectorAll('.nav-tab');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const tabId = tab.dataset.tab;
        switchTab(tabId);
      });
    });

    const dialectSelect = document.getElementById('dialectSelect');
    if (dialectSelect) {
      dialectSelect.addEventListener('change', (e) => {
        state.dialect = e.target.value;
      });
    }
  }

  function switchTab(tabId) {
    state.activeTab = tabId;
    document.querySelectorAll('.nav-tab').forEach(t => {
      t.classList.toggle('active', t.dataset.tab === tabId);
    });
    document.querySelectorAll('.app-tab').forEach(s => {
      s.classList.toggle('active', s.id === `tab-${tabId}`);
    });

    setTimeout(() => {
      dualAvatar.resize();
      studioAvatar.resize();
      dictAvatar.resize();
      dualCamera.resize();
      studioCamera.resize();
    }, 80);
  }

  // ==========================================================================
  // ONBOARDING & USER PROFILE MODAL
  // ==========================================================================
  function initOnboarding() {
    const modal = document.getElementById('onboardingModal');
    const roleCards = document.querySelectorAll('.role-card');
    const completeBtn = document.getElementById('completeOnboardingBtn');
    const userProfileBtn = document.getElementById('userProfileBtn');

    roleCards.forEach(card => {
      card.addEventListener('click', () => {
        roleCards.forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        state.userRole = card.dataset.role;
      });
    });

    completeBtn.addEventListener('click', () => {
      const nameInput = document.getElementById('onboardingUserName');
      const dialectInput = document.getElementById('onboardingDialect');
      if (nameInput) state.userName = nameInput.value || 'User';
      if (dialectInput) {
        state.dialect = dialectInput.value;
        const mainDialect = document.getElementById('dialectSelect');
        if (mainDialect) mainDialect.value = state.dialect;
      }

      updateUserRoleUI();
      modal.classList.add('hidden');

      if (state.userRole === 'deaf') {
        switchTab('signer-studio');
        studioCamera.startCamera();
      } else if (state.userRole === 'hearing') {
        switchTab('avatar-studio');
      } else {
        switchTab('dual-room');
        dualCamera.startCamera();
      }
    });

    userProfileBtn.addEventListener('click', () => {
      modal.classList.remove('hidden');
    });
  }

  function updateUserRoleUI() {
    const roleLabel = document.getElementById('userRoleName');
    const roleIcon = document.getElementById('userRoleIcon');
    if (state.userRole === 'deaf') {
      if (roleLabel) roleLabel.innerText = 'Deaf Mode';
      if (roleIcon) roleIcon.className = 'fa-solid fa-hands';
    } else if (state.userRole === 'hearing') {
      if (roleLabel) roleLabel.innerText = 'Hearing Mode';
      if (roleIcon) roleIcon.className = 'fa-solid fa-headset';
    } else {
      if (roleLabel) roleLabel.innerText = 'Dual Mode';
      if (roleIcon) roleIcon.className = 'fa-solid fa-people-arrows';
    }
  }

  // ==========================================================================
  // DUAL ROOM FUNCTIONALITY
  // ==========================================================================
  function initDualRoom() {
    const cameraToggle = document.getElementById('dualCameraToggle');
    const micBtn = document.getElementById('dualRecordVoiceBtn');
    const sendBtn = document.getElementById('dualSendTextBtn');
    const textInput = document.getElementById('dualTextInput');
    const speakBtn = document.getElementById('dualSignerSpeakBtn');
    const simulateBtn = document.getElementById('dualSignerSimulateBtn');
    const clearChat = document.getElementById('clearChatBtn');
    const sosChips = document.querySelectorAll('.sos-chip');
    const replayAvatarBtn = document.getElementById('dualAvatarReplay');

    if (cameraToggle) {
      cameraToggle.addEventListener('click', () => {
        if (dualCamera.isRunning) {
          dualCamera.stopCamera();
          cameraToggle.classList.remove('active');
        } else {
          dualCamera.startCamera();
          cameraToggle.classList.add('active');
        }
      });
    }

    if (micBtn) {
      micBtn.addEventListener('click', () => speechEngine.toggleListening());
    }

    if (sendBtn && textInput) {
      const sendAction = () => {
        const text = textInput.value.trim();
        if (text) {
          handleHearingMessage(text);
          textInput.value = '';
        }
      };
      sendBtn.addEventListener('click', sendAction);
      textInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendAction(); });
    }

    if (speakBtn) {
      speakBtn.addEventListener('click', () => {
        const sentence = document.getElementById('dualSignerSentence').innerText.replace(/"/g, '');
        speechEngine.speak(sentence);
      });
    }

    if (simulateBtn) {
      simulateBtn.addEventListener('click', () => {
        const sampleSigns = ["HELLO", "GOOD MORNING", "WATER", "HELP", "THANK YOU", "DOCTOR", "WHERE HOSPITAL"];
        const randomSign = sampleSigns[Math.floor(Math.random() * sampleSigns.length)];
        handleDualSignPrediction({
          sign: randomSign,
          confidence: 0.96,
          top_k: [{ label: randomSign, confidence: 0.96 }]
        });
      });
    }

    if (clearChat) {
      clearChat.addEventListener('click', () => {
        const messages = document.getElementById('dualChatMessages');
        if (messages) messages.innerHTML = '';
      });
    }

    if (replayAvatarBtn) {
      replayAvatarBtn.addEventListener('click', () => dualAvatar.restart());
    }

    sosChips.forEach(chip => {
      chip.addEventListener('click', () => {
        const phrase = chip.dataset.phrase;
        handleHearingMessage(phrase);
      });
    });
  }

  async function handleHearingMessage(text) {
    appendChatMessage('Hearing Companion', text, 'right');
    try {
      const res = await fetch('/api/text_to_gloss', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text, region: state.dialect })
      });
      if (res.ok) {
        const data = await res.json();
        if (data.avatar_sequence && data.avatar_sequence.length > 0) {
          dualAvatar.loadSequence(data.avatar_sequence);
          const stateBadge = document.getElementById('dualAvatarState');
          if (stateBadge) stateBadge.innerHTML = '<i class="fa-solid fa-play"></i> Signing Now...';

          // Update sequence tracker
          const tracker = document.getElementById('dualAvatarGlossSequence');
          if (tracker && data.gloss_tokens) {
            tracker.innerHTML = data.gloss_tokens.map((g, i) => `
              <span class="gloss-seq-pill ${i === 0 ? 'active' : ''}">${g}</span>
            `).join('');
          }
        }
      }
    } catch (e) {
      console.warn("[App] text_to_gloss API error:", e);
    }
  }

  function handleDualSignPrediction(prediction) {
    if (!prediction || !prediction.sign) return;
    const sign = prediction.sign;
    const conf = Math.round((prediction.confidence || 0.9) * 100);

    const signBadge = document.getElementById('dualCurrentSign');
    const fill = document.getElementById('dualConfidenceFill');
    const trackingTag = document.getElementById('dualTrackingStatus');
    if (signBadge) signBadge.innerText = sign;
    if (fill) fill.style.width = `${conf}%`;
    if (trackingTag) trackingTag.innerHTML = `<i class="fa-solid fa-check"></i> Sign Detected`;

    // Fetch formulated sentence
    fetch('/api/gloss_to_text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gloss_tokens: [sign] })
    })
    .then(r => r.json())
    .then(data => {
      const sentence = data.natural_sentence || sign;
      const display = document.getElementById('dualSignerSentence');
      if (display) display.innerText = `"${sentence}"`;

      appendChatMessage('Deaf Signer', sentence, 'left', [sign]);

      if (state.autoTts) {
        speechEngine.speak(sentence);
      }
    });
  }

  function updateDualAvatarGlossTracker(word) {
    const tracker = document.getElementById('dualAvatarGlossSequence');
    if (!tracker) return;
    const pills = tracker.querySelectorAll('.gloss-seq-pill');
    pills.forEach(p => {
      p.classList.toggle('active', p.innerText.trim().toUpperCase() === word.toUpperCase());
    });
  }

  function appendChatMessage(speaker, text, side, tokens = []) {
    const container = document.getElementById('dualChatMessages');
    if (!container) return;

    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${side}-bubble`;

    const tokenHtml = tokens.map(t => `<span class="token-pill">${t}</span>`).join('');

    bubble.innerHTML = `
      <div class="bubble-header">
        <span class="bubble-speaker"><i class="fa-solid ${side === 'left' ? 'fa-hands' : 'fa-microphone'}"></i> ${speaker}</span>
        <span class="bubble-time">${time}</span>
      </div>
      <div class="bubble-body">
        <p>${text}</p>
        ${tokenHtml ? `<div class="bubble-tokens">${tokenHtml}</div>` : ''}
      </div>
    `;

    container.appendChild(bubble);
    container.scrollTop = container.scrollHeight;
  }

  // ==========================================================================
  // SIGNER STUDIO
  // ==========================================================================
  function initSignerStudio() {
    const startBtn = document.getElementById('cameraStartBtn');
    const skeletonToggle = document.getElementById('skeletonOverlayToggle');
    const sampleBtn = document.getElementById('sampleGestureBtn');
    const clearBtn = document.getElementById('clearBufferBtn');
    const autoSpeechToggle = document.getElementById('autoSpeechToggle');
    const speakBtn = document.getElementById('studioSpeakSentenceBtn');
    const copyBtn = document.getElementById('copySentenceBtn');

    if (startBtn) {
      startBtn.addEventListener('click', () => {
        if (studioCamera.isRunning) {
          studioCamera.stopCamera();
          startBtn.innerHTML = '<i class="fa-solid fa-play"></i> Start Camera';
          startBtn.className = 'btn-glow-green';
        } else {
          studioCamera.startCamera();
          startBtn.innerHTML = '<i class="fa-solid fa-stop"></i> Stop Camera';
          startBtn.className = 'btn-secondary-sm';
        }
      });
    }

    if (skeletonToggle) {
      skeletonToggle.addEventListener('click', () => {
        studioCamera.showSkeleton = !studioCamera.showSkeleton;
        skeletonToggle.classList.toggle('active', studioCamera.showSkeleton);
      });
    }

    if (sampleBtn) {
      sampleBtn.addEventListener('click', () => {
        const testSigns = ["HELLO", "WATER", "GOOD", "THANK YOU", "HELP", "DEAF", "DOCTOR", "WHERE", "FOOD"];
        const rand = testSigns[Math.floor(Math.random() * testSigns.length)];
        handleStudioSignPrediction({
          sign: rand,
          confidence: 0.95,
          top_k: [
            { label: rand, confidence: 0.95 },
            { label: "GOOD", confidence: 0.03 },
            { label: "HELLO", confidence: 0.02 }
          ]
        });
      });
    }

    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        studioCamera.frameBuffer = [];
        const text = document.getElementById('bufferCountText');
        const bar = document.getElementById('bufferProgressBar');
        if (text) text.innerText = '0 / 15';
        if (bar) bar.style.width = '0%';
      });
    }

    if (autoSpeechToggle) {
      autoSpeechToggle.addEventListener('change', (e) => {
        state.autoTts = e.target.checked;
      });
    }

    if (speakBtn) {
      speakBtn.addEventListener('click', () => {
        const s = document.getElementById('studioFormulatedSentence').innerText.replace(/"/g, '');
        speechEngine.speak(s);
      });
    }

    if (copyBtn) {
      copyBtn.addEventListener('click', () => {
        const s = document.getElementById('studioFormulatedSentence').innerText.replace(/"/g, '');
        navigator.clipboard.writeText(s);
        copyBtn.innerHTML = '<i class="fa-solid fa-check"></i> Copied';
        setTimeout(() => { copyBtn.innerHTML = '<i class="fa-solid fa-copy"></i> Copy'; }, 2000);
      });
    }
  }

  function handleStudioSignPrediction(pred) {
    if (!pred || !pred.sign) return;
    const sign = pred.sign;
    const conf = Math.round((pred.confidence || 0.9) * 100);

    const primarySign = document.getElementById('studioPrimarySign');
    const primaryConf = document.getElementById('studioPrimaryConfidence');
    const confLabel = document.getElementById('studioConfidenceLabel');
    if (primarySign) primarySign.innerText = sign;
    if (primaryConf) primaryConf.style.width = `${conf}%`;
    if (confLabel) confLabel.innerText = `${conf}% Neural Confidence`;

    const candidateList = document.getElementById('candidateProbabilities');
    if (candidateList && pred.top_k) {
      candidateList.innerHTML = pred.top_k.map(c => `
        <div class="candidate-row">
          <span class="cand-name">${c.label}</span>
          <div class="cand-bar-wrap"><div class="cand-bar-fill" style="width: ${Math.round(c.confidence * 100)}%;"></div></div>
          <span class="cand-val">${c.confidence.toFixed(2)}</span>
        </div>
      `).join('');
    }

    fetch('/api/gloss_to_text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gloss_tokens: [sign] })
    })
    .then(r => r.json())
    .then(data => {
      const sentence = data.natural_sentence || sign;
      const sentenceBox = document.getElementById('studioFormulatedSentence');
      const glossTags = document.getElementById('studioGlossTokens');
      if (sentenceBox) sentenceBox.innerText = `"${sentence}"`;
      if (glossTags) glossTags.innerHTML = `<span class="token-pill">${sign}</span>`;

      if (state.autoTts) {
        speechEngine.speak(sentence);
      }
    });
  }

  // ==========================================================================
  // AVATAR STUDIO
  // ==========================================================================
  function initAvatarStudio() {
    const playPauseBtn = document.getElementById('avatarPlayPauseBtn');
    const restartBtn = document.getElementById('avatarRestartBtn');
    const speedSelect = document.getElementById('avatarStudioSpeed');
    const skinBtns = document.querySelectorAll('.skin-btn');
    const micRecordBtn = document.getElementById('voiceRecordButton');
    const translateBtn = document.getElementById('avatarTranslateBtn');
    const textInput = document.getElementById('avatarTextInput');
    const phrasePills = document.querySelectorAll('.phrase-pill');

    skinBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        skinBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        studioAvatar.setStyle(btn.dataset.skin);
      });
    });

    if (playPauseBtn) {
      playPauseBtn.addEventListener('click', () => {
        if (studioAvatar.isPlaying) {
          studioAvatar.pause();
          playPauseBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
        } else {
          studioAvatar.resume();
          playPauseBtn.innerHTML = '<i class="fa-solid fa-pause"></i>';
        }
      });
    }

    if (restartBtn) {
      restartBtn.addEventListener('click', () => studioAvatar.restart());
    }

    if (speedSelect) {
      speedSelect.addEventListener('change', (e) => {
        studioAvatar.setSpeed(parseFloat(e.target.value));
      });
    }

    if (micRecordBtn) {
      micRecordBtn.addEventListener('click', () => speechEngine.toggleListening());
    }

    if (translateBtn && textInput) {
      const executeTranslation = () => {
        const text = textInput.value.trim();
        if (text) translateAndPlayStudioAvatar(text);
      };
      translateBtn.addEventListener('click', executeTranslation);
      textInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') executeTranslation(); });
    }

    phrasePills.forEach(pill => {
      pill.addEventListener('click', () => {
        const phrase = pill.dataset.phrase;
        if (textInput) textInput.value = phrase;
        translateAndPlayStudioAvatar(phrase);
      });
    });

    // Load initial greeting demonstration
    translateAndPlayStudioAvatar("Hello, nice to meet you");
  }

  async function translateAndPlayStudioAvatar(text) {
    try {
      const res = await fetch('/api/text_to_gloss', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text, region: state.dialect })
      });
      if (res.ok) {
        const data = await res.json();
        if (data.avatar_sequence && data.avatar_sequence.length > 0) {
          studioAvatar.loadSequence(data.avatar_sequence);

          // Populate sequence chips
          const chipsContainer = document.getElementById('studioSequenceChips');
          if (chipsContainer && data.gloss_tokens) {
            chipsContainer.innerHTML = data.gloss_tokens.map((token, idx) => `
              <div class="seq-chip ${idx === 0 ? 'active' : ''}" data-index="${idx}">
                <span class="chip-idx">${idx + 1}</span>
                <span class="chip-word">${token}</span>
              </div>
            `).join('');
          }
        }
      }
    } catch (e) {
      console.warn("[App] translateAndPlayStudioAvatar error:", e);
    }
  }

  function highlightStudioChip(index, word) {
    const badge = document.getElementById('currentSigningWordBadge');
    if (badge) badge.innerText = `WORD: ${word}`;

    const chips = document.querySelectorAll('#studioSequenceChips .seq-chip');
    chips.forEach((c, idx) => {
      c.classList.toggle('active', idx === index);
    });
  }

  function handleVoiceInput(text, isFinal) {
    const avatarInput = document.getElementById('avatarTextInput');
    const dualInput = document.getElementById('dualTextInput');
    if (avatarInput) avatarInput.value = text;
    if (dualInput) dualInput.value = text;

    if (isFinal && text.trim().length > 0) {
      if (state.activeTab === 'avatar-studio') {
        translateAndPlayStudioAvatar(text);
      } else if (state.activeTab === 'dual-room') {
        handleHearingMessage(text);
      }
    }
  }

  function updateMicUI(isListening) {
    const micOrb = document.getElementById('voiceRecordButton');
    const micLabel = document.getElementById('micStatusLabel');
    const waveWrap = document.getElementById('audioWaveWrap');
    const dualMic = document.getElementById('dualRecordVoiceBtn');

    if (isListening) {
      if (micOrb) micOrb.classList.add('recording');
      if (micLabel) micLabel.innerText = 'Listening to your voice...';
      if (waveWrap) waveWrap.classList.add('active');
      if (dualMic) dualMic.classList.add('recording');
    } else {
      if (micOrb) micOrb.classList.remove('recording');
      if (micLabel) micLabel.innerText = 'Click to Start Speaking';
      if (waveWrap) waveWrap.classList.remove('active');
      if (dualMic) dualMic.classList.remove('recording');
    }
  }

  // ==========================================================================
  // ISL DICTIONARY & LEARNING HUB
  // ==========================================================================
  async function initDictionary() {
    const searchInput = document.getElementById('dictSearchInput');
    const catPills = document.querySelectorAll('#dictCategoryFilters .cat-pill');
    const replayBtn = document.getElementById('dictReplayBtn');
    const practiceBtn = document.getElementById('startPracticeBtn');

    try {
      const res = await fetch('/api/dictionary');
      if (res.ok) {
        state.dictionaryData = await res.json();
        renderDictionaryCards('all');
        playDictionaryPreview('HELLO');
      }
    } catch (e) {
      console.warn("[App] Load dictionary error:", e);
    }

    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        filterDictionary(query);
      });
    }

    catPills.forEach(pill => {
      pill.addEventListener('click', () => {
        catPills.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        renderDictionaryCards(pill.dataset.cat);
      });
    });

    if (replayBtn) {
      replayBtn.addEventListener('click', () => dictAvatar.restart());
    }

    if (practiceBtn) {
      practiceBtn.addEventListener('click', () => {
        switchTab('signer-studio');
        studioCamera.startCamera();
      });
    }
  }

  function renderDictionaryCards(category = 'all') {
    const grid = document.getElementById('signsCatalogGrid');
    if (!grid || !state.dictionaryData) return;

    let items = [];
    if (category === 'all') {
      Object.entries(state.dictionaryData.categories).forEach(([cat, list]) => {
        list.forEach(item => items.push({ ...item, category: cat }));
      });
    } else {
      items = (state.dictionaryData.categories[category] || []).map(item => ({ ...item, category }));
    }

    grid.innerHTML = items.map(item => `
      <div class="sign-catalog-card ${item.word === state.activeDictWord ? 'active' : ''}" data-word="${item.word}">
        <div class="card-top">
          <span class="card-category">${item.category}</span>
          <span class="card-frames-badge">${item.frames_count} Frames</span>
        </div>
        <div class="card-word">${item.word}</div>
        <div class="card-meaning">${item.meaning || 'ISL sign gesture'}</div>
      </div>
    `).join('');

    grid.querySelectorAll('.sign-catalog-card').forEach(card => {
      card.addEventListener('click', () => {
        grid.querySelectorAll('.sign-catalog-card').forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        const word = card.dataset.word;
        playDictionaryPreview(word);
      });
    });
  }

  function filterDictionary(query) {
    const cards = document.querySelectorAll('#signsCatalogGrid .sign-catalog-card');
    cards.forEach(card => {
      const word = card.dataset.word.toLowerCase();
      card.style.display = word.includes(query) ? 'flex' : 'none';
    });
  }

  async function playDictionaryPreview(word) {
    state.activeDictWord = word;
    const title = document.getElementById('dictPreviewWord');
    const desc = document.getElementById('dictSignDescription');
    if (title) title.innerText = word;

    try {
      const res = await fetch(`/api/sign_keyframes/${word}?region=${state.dialect}`);
      if (res.ok) {
        const signData = await res.json();
        dictAvatar.playSingleSign(signData);
        if (desc) desc.innerText = signData.meaning || `ISL execution for '${word}'`;
      }
    } catch (e) {
      console.warn("[App] playDictionaryPreview error:", e);
    }
  }

  // ==========================================================================
  // SETTINGS MODAL
  // ==========================================================================
  function initSettings() {
    const toggleBtn = document.getElementById('settingsToggleBtn');
    const modal = document.getElementById('settingsModal');
    const closeBtn = document.getElementById('closeSettingsBtn');
    const saveBtn = document.getElementById('saveSettingsBtn');
    const autoTtsCheckbox = document.getElementById('settingAutoTts');
    const ttsRateSlider = document.getElementById('settingTtsRate');
    const skeletonCheckbox = document.getElementById('settingShowSkeleton');

    if (toggleBtn && modal) {
      toggleBtn.addEventListener('click', () => modal.classList.remove('hidden'));
    }
    if (closeBtn && modal) {
      closeBtn.addEventListener('click', () => modal.classList.add('hidden'));
    }
    if (saveBtn && modal) {
      saveBtn.addEventListener('click', () => {
        if (autoTtsCheckbox) state.autoTts = autoTtsCheckbox.checked;
        if (ttsRateSlider) {
          state.ttsRate = parseFloat(ttsRateSlider.value);
          speechEngine.ttsRate = state.ttsRate;
        }
        if (skeletonCheckbox) {
          state.showSkeleton = skeletonCheckbox.checked;
          dualCamera.showSkeleton = state.showSkeleton;
          studioCamera.showSkeleton = state.showSkeleton;
        }
        modal.classList.add('hidden');
      });
    }
  }

  async function checkSystemStatus() {
    const badge = document.getElementById('systemStatusBadge');
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        if (badge) {
          badge.className = 'system-status online';
          badge.title = `Classes: ${data.classes_count}, Models: Ready`;
        }
      }
    } catch (e) {
      if (badge) {
        badge.className = 'system-status offline';
        badge.innerHTML = '<span class="status-dot"></span><span>AI Standby</span>';
      }
    }
  }
});
