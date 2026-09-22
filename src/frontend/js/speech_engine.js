/**
 * Speech Engine: Web Speech API wrapper for both TTS (Synthesis) and STT (Recognition)
 * Features:
 * - Offline-capable browser speech synthesis
 * - Real-time microphone dictation
 * - Audio visualizer wave generator
 */

class SpeechEngine {
  constructor(options = {}) {
    this.ttsRate = options.ttsRate || 1.0;
    this.ttsPitch = options.ttsPitch || 1.0;
    this.autoSpeak = options.autoSpeak !== undefined ? options.autoSpeak : true;

    this.isListening = false;
    this.recognition = null;
    this.onSpeechResult = options.onSpeechResult || null;
    this.onListeningChange = options.onListeningChange || null;

    this._initSTT();
  }

  _initSTT() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.warn("[SpeechEngine] Web SpeechRecognition API is not supported in this browser.");
      return;
    }

    this.recognition = new SpeechRecognition();
    this.recognition.continuous = false;
    this.recognition.interimResults = true;
    this.recognition.lang = 'en-IN'; // Indian English dialect preference

    this.recognition.onstart = () => {
      this.isListening = true;
      if (this.onListeningChange) this.onListeningChange(true);
    };

    this.recognition.onend = () => {
      this.isListening = false;
      if (this.onListeningChange) this.onListeningChange(false);
    };

    this.recognition.onerror = (event) => {
      console.warn("[SpeechEngine] STT Error:", event.error);
      this.isListening = false;
      if (this.onListeningChange) this.onListeningChange(false);
    };

    this.recognition.onresult = (event) => {
      let interimTranscript = '';
      let finalTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          interimTranscript += event.results[i][0].transcript;
        }
      }

      if (this.onSpeechResult) {
        this.onSpeechResult(finalTranscript || interimTranscript, Boolean(finalTranscript));
      }
    };
  }

  toggleListening() {
    if (!this.recognition) {
      alert("Speech recognition is not supported in this browser. Please type your message.");
      return;
    }

    if (this.isListening) {
      this.recognition.stop();
    } else {
      try {
        this.recognition.start();
      } catch (e) {
        console.warn("[SpeechEngine] Failed to start STT:", e);
      }
    }
  }

  speak(text) {
    if (!('speechSynthesis' in window) || !text || text.trim().length === 0) return;

    window.speechSynthesis.cancel(); // Cancel any ongoing speech

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = this.ttsRate;
    utterance.pitch = this.ttsPitch;

    // Pick English or Indian English voice if available
    const voices = window.speechSynthesis.getVoices();
    const preferredVoice = voices.find(v => v.lang.includes('en-IN') || v.lang.includes('en-US')) || voices[0];
    if (preferredVoice) {
      utterance.voice = preferredVoice;
    }

    window.speechSynthesis.speak(utterance);
  }
}
