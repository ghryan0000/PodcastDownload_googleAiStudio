import React, { useState, useRef } from 'react';
import { Search, Globe, AlertCircle, UploadCloud, FileAudio, Link as LinkIcon } from 'lucide-react';
import Header from './components/Header';
import StepIndicator from './components/StepIndicator';
import ResultCard from './components/ResultCard';
import { fetchUrlSource } from './services/proxyService';
import { extractStreamUrlFromSource } from './services/geminiService';
import { AppState, StepStatus } from './types';

function App() {
  const [mode, setMode] = useState<'scan' | 'upload'>('scan');
  const [url, setUrl] = useState('');
  const [appState, setAppState] = useState<AppState>(AppState.IDLE);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [extractedStreamUrl, setExtractedStreamUrl] = useState<string | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [steps, setSteps] = useState<StepStatus[]>([
    { id: 'fetch', label: 'Fetching Source Code', status: 'pending' },
    { id: 'analyze', label: 'Gemini AI Analysis', status: 'pending' },
    { id: 'extract', label: 'Extracting Audio Stream', status: 'pending' },
  ]);

  const updateStepStatus = (id: string, status: StepStatus['status']) => {
    setSteps(prev => prev.map(step => step.id === id ? { ...step, status } : step));
  };

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url) return;

    // Reset State
    setAppState(AppState.FETCHING_SOURCE);
    setErrorMessage(null);
    setExtractedStreamUrl(null);
    setUploadedFileName(null);
    setSteps([
      { id: 'fetch', label: 'Fetching Webpage Source', status: 'loading' },
      { id: 'analyze', label: 'Gemini AI Analysis', status: 'pending' },
      { id: 'extract', label: 'Extracting Audio Stream', status: 'pending' },
    ]);

    try {
      // Step 1: Fetch Source
      updateStepStatus('fetch', 'loading');
      let sourceCode = '';
      try {
        sourceCode = await fetchUrlSource(url);
        updateStepStatus('fetch', 'completed');
      } catch (err) {
        updateStepStatus('fetch', 'error');
        throw new Error("Could not fetch the webpage. Check the URL or the site may block proxies.");
      }

      // Step 2 & 3: Analyze with Gemini
      setAppState(AppState.ANALYZING_CODE);
      updateStepStatus('analyze', 'loading');
      updateStepStatus('extract', 'loading');
      
      const streamUrl = await extractStreamUrlFromSource(sourceCode);
      
      if (!streamUrl) {
        updateStepStatus('analyze', 'completed'); // Analysis worked, just didn't find it
        updateStepStatus('extract', 'error');
        throw new Error("Gemini could not locate a 'streamURL' variable in the source code.");
      }

      updateStepStatus('analyze', 'completed');
      updateStepStatus('extract', 'completed');
      
      setExtractedStreamUrl(streamUrl);
      setAppState(AppState.SUCCESS);

    } catch (err) {
      setAppState(AppState.ERROR);
      setErrorMessage(err instanceof Error ? err.message : "An unknown error occurred.");
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('audio/')) {
      setErrorMessage("Please upload a valid audio file (MP3, WAV, etc.)");
      setAppState(AppState.ERROR);
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    setExtractedStreamUrl(objectUrl);
    setUploadedFileName(file.name);
    setAppState(AppState.SUCCESS);
    setErrorMessage(null);
  };

  const triggerFileUpload = () => {
    fileInputRef.current?.click();
  };

  const handleReset = () => {
    if (uploadedFileName && extractedStreamUrl) {
      URL.revokeObjectURL(extractedStreamUrl);
    }
    setAppState(AppState.IDLE);
    setUrl('');
    setErrorMessage(null);
    setExtractedStreamUrl(null);
    setUploadedFileName(null);
    setSteps([
      { id: 'fetch', label: 'Fetching Source Code', status: 'pending' },
      { id: 'analyze', label: 'Gemini AI Analysis', status: 'pending' },
      { id: 'extract', label: 'Extracting Audio Stream', status: 'pending' },
    ]);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const isProcessing = appState === AppState.FETCHING_SOURCE || appState === AppState.ANALYZING_CODE;

  return (
    <div className="min-h-screen bg-gradient-to-br from-rose-100/90 via-white/40 to-rose-100/90 text-gray-900 flex flex-col font-sans selection:bg-rose-200 selection:text-rose-900">
      <Header />

      <main className="flex-1 container mx-auto px-4 py-8 flex flex-col items-center">
        
        {/* Input Section */}
        <div className={`w-full max-w-2xl transition-all duration-500 ${isProcessing ? 'opacity-50 pointer-events-none blur-[1px]' : 'opacity-100'}`}>
           <div className="bg-white/95 backdrop-blur-md p-2 rounded-3xl border border-white/50 shadow-2xl shadow-rose-200/40 overflow-hidden">
              
              {/* Tabs */}
              <div className="flex p-1 gap-1 mb-2 bg-gray-200/80 rounded-2xl ring-1 ring-gray-300/10">
                <button
                  onClick={() => {
                    setMode('scan');
                    if (appState === AppState.SUCCESS) handleReset();
                  }}
                  className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-bold transition-all duration-300 transform ${
                    mode === 'scan' 
                      ? 'bg-rose-50 text-rose-900 shadow-lg shadow-rose-200/50 ring-1 ring-rose-200 hover:scale-[1.03] hover:shadow-xl hover:shadow-rose-300/40' 
                      : 'text-gray-600 hover:text-gray-900 hover:bg-white/20 hover:scale-[1.01]'
                  }`}
                >
                  <LinkIcon className={`w-4 h-4 ${mode === 'scan' ? 'text-rose-600' : ''}`} />
                  Paste URL
                </button>
                <button
                  onClick={() => {
                    setMode('upload');
                    if (appState === AppState.SUCCESS) handleReset();
                  }}
                  className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-bold transition-all duration-300 transform ${
                    mode === 'upload' 
                      ? 'bg-rose-50 text-rose-900 shadow-lg shadow-rose-200/50 ring-1 ring-rose-200 hover:scale-[1.03] hover:shadow-xl hover:shadow-rose-300/40' 
                      : 'text-gray-600 hover:text-gray-900 hover:bg-white/20 hover:scale-[1.01]'
                  }`}
                >
                  <UploadCloud className={`w-4 h-4 ${mode === 'upload' ? 'text-rose-600' : ''}`} />
                  Upload File
                </button>
              </div>

              <div className="p-4 sm:p-6">
                {mode === 'scan' ? (
                  <form onSubmit={handleAnalyze} className="relative animate-in fade-in slide-in-from-left-4 duration-300">
                    <label className="block text-xs font-bold text-gray-500 uppercase tracking-wider mb-2 ml-1">
                      Target Website URL
                    </label>
                    <div className="relative group">
                      <div className="relative flex items-center bg-gray-50 rounded-2xl overflow-hidden border border-gray-200 focus-within:border-rose-500 focus-within:ring-4 focus-within:ring-rose-500/10 transition-all">
                        <Globe className="ml-4 w-5 h-5 text-gray-400" />
                        <input
                          type="url"
                          required
                          placeholder="https://example.com/podcast-page"
                          className="w-full bg-transparent border-none focus:ring-0 text-gray-900 placeholder-gray-400 py-4 px-4 h-14 outline-none font-medium"
                          value={url}
                          onChange={(e) => setUrl(e.target.value)}
                        />
                        <button 
                          type="submit"
                          disabled={!url || isProcessing}
                          className="mr-2 px-6 py-2.5 bg-rose-500 hover:bg-rose-600 text-white font-bold rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 shadow-lg shadow-rose-500/20 active:scale-95"
                        >
                          <Search className="w-4 h-4" />
                          {appState === AppState.SUCCESS ? 'Rescan' : 'Scan'}
                        </button>
                      </div>
                    </div>
                    <p className="text-xs text-gray-400 mt-3 ml-2">
                      Supports pages where <code>streamURL</code> is defined in the source script.
                    </p>
                  </form>
                ) : (
                  <div className="animate-in fade-in slide-in-from-right-4 duration-300">
                    <label className="block text-xs font-bold text-gray-500 uppercase tracking-wider mb-2 ml-1">
                      Upload Audio File
                    </label>
                    <div 
                      onClick={triggerFileUpload}
                      className="border-2 border-dashed border-gray-200 hover:border-rose-400 hover:bg-rose-50/50 rounded-2xl p-10 flex flex-col items-center justify-center cursor-pointer group transition-all"
                    >
                      <input 
                        type="file" 
                        ref={fileInputRef}
                        onChange={handleFileUpload}
                        accept="audio/*"
                        className="hidden" 
                      />
                      <div className="w-16 h-16 bg-white shadow-lg rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform text-rose-500">
                        <FileAudio className="w-8 h-8" />
                      </div>
                      <h3 className="text-lg font-bold text-gray-900 mb-1 group-hover:text-rose-600 transition-colors">
                        Click to Upload MP3
                      </h3>
                      <p className="text-gray-400 text-sm font-medium">
                        Supports MP3, WAV, M4A
                      </p>
                    </div>
                  </div>
                )}
              </div>
           </div>
        </div>

        {/* Status Section */}
        {isProcessing && mode === 'scan' && (
          <div className="w-full animate-in fade-in slide-in-from-bottom-4 duration-500">
            <StepIndicator steps={steps} />
          </div>
        )}

        {/* Error Message */}
        {appState === AppState.ERROR && errorMessage && (
          <div className="max-w-2xl w-full bg-red-50 border border-red-100 text-red-800 p-5 rounded-2xl flex items-center gap-4 animate-in shake my-6 shadow-sm">
            <div className="bg-red-100 p-2 rounded-full">
               <AlertCircle className="w-6 h-6 shrink-0 text-red-600" />
            </div>
            <div className="flex-1">
              <p className="font-bold text-red-900">Operation Failed</p>
              <p className="text-sm opacity-90">{errorMessage}</p>
            </div>
            <button 
              onClick={handleReset} 
              className="px-5 py-2.5 bg-white hover:bg-red-50 border border-red-200 text-red-700 rounded-xl text-sm font-bold transition-colors shadow-sm"
            >
              Try Again
            </button>
          </div>
        )}

        {/* Success Result */}
        {appState === AppState.SUCCESS && extractedStreamUrl && (
          <div className="animate-in zoom-in-95 duration-500 w-full flex justify-center">
            <ResultCard 
              streamUrl={extractedStreamUrl} 
              onReset={handleReset} 
              isUpload={mode === 'upload'}
              fileName={uploadedFileName}
            />
          </div>
        )}

      </main>

      <footer className="w-full py-8 text-center text-gray-400 text-sm">
        <p className="font-medium">Powered by Gemini 1.5 Flash • Uses allorigins.win for proxy</p>
      </footer>
    </div>
  );
}

export default App;