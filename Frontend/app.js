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
        // Drafter model (Gemini) is always available in backend, so can be enabled if not executing
        drafterModelSelect.disabled = !enabled;
        // Target model (Ollama) depends on backend availability, but generally controlled here
        // Enable only if not executing AND there are actual models in the dropdown
        targetModelSelect.disabled = !enabled || targetModelSelect.options.length === 0 || targetModelSelect.options[0].value === 'No models found' || targetModelSelect.options[0].value === 'Connection failed';
        
        refreshModelsBtn.disabled = !enabled;
        // Export button is disabled if not enabled or no content
        exportBtn.disabled = !enabled || fullReportContent === '';
    };
    
    // --- Model Population Functions ---
    const populateDrafterModels = () => {
        drafterModelSelect.innerHTML = '';
        // The backend's "Mind" (Gemini) is hardcoded to Gemini 1.5 Flash.
        // We only offer this option to reflect backend's behavior.
        const modelName = "Gemini 1.5 Flash"; 
        const option = document.createElement('option');
        option.value = modelName;
        option.textContent = modelName;
        drafterModelSelect.appendChild(option);
        drafterModelSelect.value = modelName; // Ensure it's selected
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
            
            targetModelSelect.innerHTML = ''; // Clear existing options
            
            // --- CORRECTED LOGIC HERE ---
            // The backend's /api/ollama/models endpoint proxies Ollama's /api/tags,
            // which returns an object with a 'models' array.
            if (data.models && data.models.length > 0) {
                // Optional: Add a "Select a model" default option if desired
                // const defaultOption = document.createElement('option');
                // defaultOption.value = ''; // Empty value for default
                // defaultOption.textContent = 'Select Ghostwriter Model';
                // targetModelSelect.appendChild(defaultOption);

                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.name; // Use model.name (e.g., "llama3:latest") as the option value
                    option.textContent = model.name; // Display the model name
                    targetModelSelect.appendChild(option);
                });
                // Select the first model in the list by default, or the default option if added
                targetModelSelect.value = data.models[0].name; 
                setStatus('Ready. Select models and enter a query.');
            } else {
                // This branch handles cases where 'data.models' is empty or not present
                targetModelSelect.innerHTML = '<option value="No models found">No models found</option>'; // Add a value
                setStatus('Ollama found, but no models available.', true);
            }
        } catch (error) {
            console.error('Error fetching models:', error);
            setStatus(`Could not connect to Ollama server or no models: ${error.message}`, true);
            targetModelSelect.innerHTML = '<option value="Connection failed">Connection failed</option>'; // Add a value
        } finally {
            refreshModelsBtn.classList.remove('spin'); // Remove spin animation
            toggleControls(true); // Re-enable controls after fetch attempt
        }
    };

    // --- Core Logic Functions ---
    const executeAgent = async () => {
        const query = queryInput.value.trim();
        if (!query || isExecuting) return;

        // Check if a Ghostwriter model is actually selected/available
        if (targetModelSelect.options.length === 0 || targetModelSelect.value === 'No models found' || targetModelSelect.value === 'Connection failed') {
            setStatus("Cannot execute: No Ghostwriter (Ollama) model selected or available.", true);
            return;
        }

        logOutput.innerHTML = ''; // Clear previous logs
        fullReportContent = ''; // Reset report content
        toggleControls(false); // Disable controls during execution
        setStatus('Agent is running...');
        addLog('Connecting to agent...');

        try {
            const response = await fetch('/api/agent/execute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                // CRITICAL FIX: The backend's AgentRequest Pydantic model
                // only expects 'query'. 'target_model' and 'drafter_model'
                // are determined by the backend's ResourceManager.
                body: JSON.stringify({ query: query }) 
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
                            case 'complete': 
                                setStatus('Agent finished. Report generated. You can now export it.'); 
                                // Display saved file path if available
                                if (data.file_path) {
                                    addLog(`Report saved on server at: ${data.file_path}`, 'success');
                                }
                                break;
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
        // Sanitize filename for download
        const sanitizedTitle = reportTitle.replace(/[^a-z0-9\s-]/gi, '').trim().replace(/\s+/g, '_').toLowerCase();
        a.download = `${sanitizedTitle || 'generated_report'}.html`; // Fallback if sanitized title is empty
        a.click();
        URL.revokeObjectURL(url);
    });

    // --- Initial Page Load ---
    const initialize = async () => {
        toggleControls(false); // Disable initially until models are fetched
        populateDrafterModels(); // Populate Gemini model
        await fetchOllamaModels(); // Fetch Ollama model and re-enable controls
    };

    initialize();
});
