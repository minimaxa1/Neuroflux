// --- START OF FILE app.js ---

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element Selection ---
    const statusDiv = document.getElementById('status');
    const targetModelSelect = document.getElementById('target-model-select');
    const drafterModelSelect = document.getElementById('drafter-model-select');
    const refreshModelsBtn = document.getElementById('refresh-models-btn');
    const indexBtn = document.getElementById('index-btn');
    const queryInput = document.getElementById('query-input');
    const executeBtn = document.getElementById('execute-btn');
    const exportBtn = document.getElementById('export-btn');
    const logOutput = document.getElementById('log-output');

    // --- State Variables ---
    let isExecuting = false;
    let fullReportContent = ''; // Still holds the report content for export

    // --- UI Update Functions ---
    const setStatus = (message, isError = false) => {
        statusDiv.textContent = message;
        // Use new CSS variable for error color
        statusDiv.style.color = isError ? 'var(--error-color)' : 'var(--text-color-subtle)'; 
    };

    const addLog = (content, className = '') => {
        const entry = document.createElement('div');
        entry.textContent = content;
        entry.className = `log-entry ${className}`;
        logOutput.appendChild(entry);
        logOutput.scrollTop = logOutput.scrollHeight; // Auto-scroll to bottom
    };
    
    const addJsonLog = (planObject) => {
        const entry = document.createElement('div');
        entry.className = 'log-entry plan'; // Use 'plan' class for JSON display
        entry.textContent = JSON.stringify(planObject, null, 2);
        logOutput.appendChild(entry);
        logOutput.scrollTop = logOutput.scrollHeight;
    }

    // Toggles button and input states based on whether an operation is active
    const toggleControls = (enabled) => {
        isExecuting = !enabled;
        const queryIsEmpty = queryInput.value.trim() === '';
        
        executeBtn.disabled = !enabled || queryIsEmpty;
        indexBtn.disabled = !enabled; // Index button disabled while any process (execute or index) is running
        queryInput.disabled = !enabled;
        targetModelSelect.disabled = !enabled;
        drafterModelSelect.disabled = !enabled;
        refreshModelsBtn.disabled = !enabled;
        // Export button is disabled if not enabled or no content
        exportBtn.disabled = !enabled || fullReportContent === '';
    };
    
    // --- Model Population Functions ---
    const populateDrafterModels = () => {
        drafterModelSelect.innerHTML = '';
        // These are fixed as per your Mind configuration in main.py
        const models = ["Gemini 1.5 Flash", "Gemini 1.5 Pro"]; 
        models.forEach(modelName => {
            const option = document.createElement('option');
            option.value = modelName;
            option.textContent = modelName;
            drafterModelSelect.appendChild(option);
        });
    };

    const fetchOllamaModels = async () => {
        try {
            setStatus('Fetching Ollama models...');
            refreshModelsBtn.classList.add('spin'); // Add spin animation
            const response = await fetch('/api/ollama/models');
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(`HTTP error! status: ${response.status} - ${errorData.detail || response.statusText}`);
            }
            const data = await response.json();
            
            targetModelSelect.innerHTML = '';
            if (data.models && data.models.length > 0) {
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.name;
                    option.textContent = model.name;
                    targetModelSelect.appendChild(option);
                });
                // Select 'mistral:latest' if available, otherwise the first one
                if (Array.from(targetModelSelect.options).some(option => option.value === 'mistral:latest')) {
                    targetModelSelect.value = 'mistral:latest';
                } else if (data.models.length > 0) {
                    targetModelSelect.value = data.models[0].name;
                }
                setStatus('Ready. Select models and enter a query.');
            } else {
                targetModelSelect.innerHTML = '<option>No models found</option>';
                setStatus('Ollama found, but no models available.', true);
            }
        } catch (error) {
            console.error('Error fetching models:', error);
            setStatus(`Could not connect to Ollama server or no models: ${error.message}`, true);
            targetModelSelect.innerHTML = '<option>Connection failed</option>';
        } finally {
            refreshModelsBtn.classList.remove('spin'); // Remove spin animation
        }
    };

    // --- Core Logic Functions ---
    const executeAgent = async () => {
        const query = queryInput.value.trim();
        if (!query || isExecuting) return;

        logOutput.innerHTML = ''; // Clear previous logs
        fullReportContent = ''; // Reset report content
        toggleControls(false); // Disable controls during execution
        setStatus('Agent is running...');
        addLog('Connecting to agent...');

        try {
            const response = await fetch('/api/agent/execute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query,
                    target_model: targetModelSelect.value, // This is Ollama model (Ghostwriter)
                    drafter_model: drafterModelSelect.value // This is Gemini model (Mind)
                })
            });

            if (!response.ok) { // Check for non-2xx HTTP responses
                const errorData = await response.json();
                throw new Error(`Server Error: ${errorData.detail || response.statusText}`);
            }

            if (!response.body) throw new Error("Response has no body.");
            addLog('Connection established. Streaming agent thought process...');
            
            const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
            
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                const lines = value.split('\n\n').filter(line => line.startsWith('data:'));
                for (const line of lines) {
                    const jsonData = line.substring(5);
                    try {
                        const data = JSON.parse(jsonData);
                        switch (data.type) {
                            case 'log': addLog(data.content, data.class || ''); break;
                            case 'plan_generated':
                                addLog('Received research and narrative plan:', 'phase-title');
                                addJsonLog(data.plan);
                                break;
                            case 'synthesis_token':
                                // Accumulate report content for export
                                fullReportContent += data.content; 
                                break;
                            case 'error':
                                addLog(data.content, 'error');
                                setStatus(`Agent Error: ${data.content}`, true); // Show error from backend
                                break;
                            case 'complete': setStatus('Agent finished. Report generated. You can now export it.'); break;
                        }
                    } catch (e) {
                         console.warn("Could not parse JSON from stream:", jsonData, e);
                    }
                }
            }

        } catch (error) {
            console.error('Agent execution error:', error);
            addLog(`Execution Error: ${error.message}`, 'error');
            setStatus(`Agent execution failed: ${error.message}`, true);
        } finally {
            addLog('Stream closed. Agent process finished.');
            toggleControls(true); // Re-enable controls
        }
    };

    const startIndexing = async () => {
        if (isExecuting) return; // Prevent re-triggering if any operation is active
        toggleControls(false); // Disable controls during indexing
        setStatus('Indexing knowledge base (this may take a long time)...');
        addLog('Initiating indexing process...');

        try {
            const response = await fetch('/api/rag/start-indexing', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
                // No body needed for this endpoint
            });

            const data = await response.json(); // Backend sends JSON response
            if (response.ok) {
                addLog(`Indexing initiated: ${data.message}`, 'info');
                setStatus('Indexing started in background. See server logs for progress.');
            } else {
                addLog(`Indexing failed: ${data.detail || data.message}`, 'error');
                setStatus(`Indexing failed: ${data.detail || data.message}`, true);
            }
        }
        catch (error) {
            console.error('Indexing API error:', error);
            addLog(`Indexing request failed: ${error.message}`, 'error');
            setStatus(`Indexing request failed: ${error.message}`, true);
        } finally {
            toggleControls(true); // Re-enable controls once the API call response is received
        }
    };

    // --- Initialization and Event Listeners ---
    refreshModelsBtn.addEventListener('click', fetchOllamaModels);
    executeBtn.addEventListener('click', executeAgent);
    queryInput.addEventListener('input', () => {
        if (!isExecuting) { // Only update if not currently executing
            executeBtn.disabled = queryInput.value.trim() === '';
        }
    });
    
    // Attach the startIndexing function to the indexBtn click event
    indexBtn.addEventListener('click', startIndexing);

    exportBtn.addEventListener('click', () => {
        if (!fullReportContent) return;
        const blob = new Blob([fullReportContent], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        
        // Dynamically set filename based on the report's title from the content, or default
        let reportTitle = 'synthesized_report';
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = fullReportContent;
        const titleElement = tempDiv.querySelector('head title'); // Look for the <title> tag
        if (titleElement) {
            reportTitle = titleElement.textContent;
        }
        const sanitizedTitle = reportTitle.replace(/[^a-z0-9]/gi, '_').toLowerCase();
        a.download = `${sanitizedTitle}.html`;
        a.click();
        URL.revokeObjectURL(url);
    });

    // --- Initial Page Load ---
    const initialize = async () => {
        toggleControls(false); // Disable initially until models are fetched
        populateDrafterModels();
        await fetchOllamaModels();
        toggleControls(true); // Re-enable after initialization
    };

    initialize();
});