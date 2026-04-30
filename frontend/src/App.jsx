import React, { useState, useEffect, useRef } from 'react';
import './index.css';

const SUPPORTED_LANGUAGES = [
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'zh', name: 'Mandarin' },
  { code: 'ja', name: 'Japanese' },
  { code: 'ko', name: 'Korean' },
  { code: 'vi', name: 'Vietnamese' },
  { code: 'uk', name: 'Ukrainian' },
  { code: 'ar', name: 'Arabic' },
];

const Typewriter = ({ text }) => {
  const [displayedText, setDisplayedText] = useState('');

  useEffect(() => {
    setDisplayedText('');
    if (!text) return;

    let i = 0;
    const interval = setInterval(() => {
      setDisplayedText(text.substring(0, i + 1));
      i++;
      if (i >= text.length) clearInterval(interval);
    }, 20);

    return () => clearInterval(interval);
  }, [text]);

  return <span>{displayedText}</span>;
};

function App() {
  const [speakerLang, setSpeakerLang] = useState('en');
  const [targetLangs, setTargetLangs] = useState(['Spanish']);
  const [isRecording, setIsRecording] = useState(false);
  const [chunks, setChunks] = useState([]);
  
  const wsRef = useRef(null);
  const scrollIntervalRef = useRef(null);
  const containerRefs = useRef([]);

  // Auto-scroll logic
  useEffect(() => {
    scrollIntervalRef.current = setInterval(() => {
      containerRefs.current.forEach(ref => {
        if (ref) ref.scrollTop = ref.scrollHeight;
      });
    }, 100);
    return () => clearInterval(scrollIntervalRef.current);
  }, []);

  useEffect(() => {
    // Connect to WebSocket
    wsRef.current = new WebSocket('ws://localhost:8000/ws');
    
    wsRef.current.onopen = () => {
      console.log('WebSocket Connected');
    };
    
    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'transcription_start') {
        const initialTranslations = {};
        if (data.target_langs) {
          data.target_langs.forEach(lang => {
            initialTranslations[lang] = '';
          });
        }
        
        setChunks(prev => {
          const newChunk = {
            id: Date.now() + Math.random(),
            source_text: data.source_text,
            speaker_lang: data.speaker_lang,
            translations: initialTranslations
          };
          const updated = [...prev, newChunk];
          // Limit history so we don't leak memory, 1000 chunks is plenty for scrolling
          if (updated.length > 1000) return updated.slice(updated.length - 1000);
          return updated;
        });
      } else if (data.type === 'translation_update') {
        setChunks(prev => {
          if (prev.length === 0) return prev;
          const newChunks = [...prev];
          // Update the specific chunk that matches the source text
          for (let i = newChunks.length - 1; i >= 0; i--) {
            if (newChunks[i].source_text === data.source_text) {
              newChunks[i] = {
                ...newChunks[i],
                translations: {
                  ...newChunks[i].translations,
                  [data.lang]: data.text
                }
              };
              break;
            }
          }
          return newChunks;
        });
      }
    };
    
    wsRef.current.onerror = (error) => {
      console.error('WebSocket Error:', error);
    };

    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const handleToggleRecording = () => {
    if (isRecording) {
      wsRef.current.send(JSON.stringify({ action: 'stop' }));
    } else {
      setChunks([]);
      wsRef.current.send(JSON.stringify({
        action: 'start',
        speaker_lang: speakerLang,
        target_langs: targetLangs
      }));
    }
    setIsRecording(!isRecording);
  };

  const handleTargetLangToggle = (langName) => {
    // Enforce single selection to prevent GPU overload
    setTargetLangs([langName]);
  };

  // When changing speaker language or target languages while recording, update the backend
  useEffect(() => {
    if (isRecording && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        action: 'start',
        speaker_lang: speakerLang,
        target_langs: targetLangs
      }));
    }
  }, [speakerLang, targetLangs]);

  return (
    <div className="app-container">
      <div className="background-shapes">
        <div className="shape shape-1"></div>
        <div className="shape shape-2"></div>
        <div className="shape shape-3"></div>
      </div>
      
      <header className="glass-header">
        <div className="header-main-row">
          <h1>Live<span>Translate</span></h1>
          
          <div className="controls-inline">
            <div className="speaker-select-wrapper">
              <span className="inline-label">Speaker:</span>
              <select 
                value={speakerLang} 
                onChange={(e) => setSpeakerLang(e.target.value)}
                className="glass-select inline-select"
              >
                {SUPPORTED_LANGUAGES.map(lang => (
                  <option key={lang.code} value={lang.code}>{lang.name}</option>
                ))}
              </select>
            </div>
            
            <div className="divider"></div>
            
            <div className="target-languages-inline">
              <span className="inline-label">Targets:</span>
              <div className="chip-container horizontal-scroll">
                {SUPPORTED_LANGUAGES.map(lang => {
                  const isSelected = targetLangs.includes(lang.name);
                  return (
                    <div 
                      key={lang.name} 
                      className={`chip ${isSelected ? 'selected' : ''}`}
                      onClick={() => handleTargetLangToggle(lang.name)}
                    >
                      {lang.name}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <button 
            className={`record-button ${isRecording ? 'recording' : ''}`}
            onClick={handleToggleRecording}
          >
            <div className="record-indicator"></div>
            {isRecording ? 'Listening...' : 'Start'}
          </button>
        </div>
      </header>

      <main className="content">
        {chunks.length === 0 && !isRecording && (
          <div className="empty-state">
            <h2>Ready to translate</h2>
            <p>Select your languages and press Start to begin speaking.</p>
          </div>
        )}
        
        {(chunks.length > 0 || isRecording) && (
          <div className="channels-container">
            <div className="channel-box source-channel glass-panel">
              <span className="lang-badge source-badge">{speakerLang.toUpperCase()}</span>
              <div 
                className="channel-text-container"
                ref={el => containerRefs.current[0] = el}
              >
                {chunks.map(chunk => (
                  <span key={chunk.id} className="chunk-span">
                    <Typewriter text={chunk.source_text} />
                  </span>
                ))}
              </div>
            </div>
            
            <div className="translations-grid">
              {targetLangs.map((lang, idx) => (
                <div key={lang} className="channel-box target-channel glass-panel">
                  <span className="lang-badge target-badge">{lang.toUpperCase()}</span>
                  <div 
                    className="channel-text-container"
                    ref={el => containerRefs.current[idx + 1] = el}
                  >
                    {chunks.map((chunk, chunkIdx) => {
                      const text = chunk.translations[lang];
                      const isLast = chunkIdx === chunks.length - 1;
                      
                      return (
                        <span key={chunk.id} className="chunk-span">
                          {text ? <Typewriter text={text} /> : (isLast ? <span style={{ opacity: 0.5 }}>Translating...</span> : null)}
                        </span>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
