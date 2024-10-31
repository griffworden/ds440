document.addEventListener('DOMContentLoaded', function() {
    const textInput = document.getElementById('news-input');
    const analyzeButton = document.getElementById('analyze-text');
    const clearButton = document.getElementById('clear-text');
    const resultContainer = document.getElementById('result-container');
    const loading = document.getElementById('loading');
    const charCount = document.getElementById('char-count');

    // Auto-inject fact checker for Twitter/X pages
    chrome.tabs.query({active: true, currentWindow: true}, async function(tabs) {
        const currentTab = tabs[0];
        if (currentTab.url.includes("twitter.com") || currentTab.url.includes("x.com")) {
            try {
                await chrome.scripting.executeScript({
                    target: { tabId: currentTab.id },
                    files: ['inject.js']
                });
                
                await chrome.scripting.executeScript({
                    target: { tabId: currentTab.id },
                    function: () => {
                        window.injectFactChecker();
                    }
                });
            } catch (error) {
                console.error('Error injecting fact checker:', error);
            }
        }
    });

    // Character count update
    textInput.addEventListener('input', () => {
        const length = textInput.value.length;
        charCount.textContent = `${length}/250 characters`;
        charCount.style.color = length > 250 ? '#D22B2B' : '#666';
    });

    // Clear functionality
    clearButton.addEventListener('click', () => {
        textInput.value = '';
        resultContainer.innerHTML = '';
        charCount.textContent = '0/250 characters';
        charCount.style.color = '#666';
    });

    function createModelBadge(modelName, isFake) {
        return `
            <div class="model-badge ${isFake ? 'fake' : 'real'}">
                ${modelName}: ${isFake ? 'FAKE' : 'REAL'}
            </div>
        `;
    }

    // Analyze text functionality
    analyzeButton.addEventListener('click', async () => {
        const text = textInput.value.trim();
        if (!text) {
            alert('Please enter some text to analyze');
            return;
        }

        loading.style.display = 'block';
        resultContainer.innerHTML = '';

        try {
            const response = await fetch('http://localhost:8000/classify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            const result = data[0];
            const isFake = result.Ensemble;
            
            resultContainer.innerHTML = `
                <div class="fact-check-result ${isFake ? 'fake' : 'real'}">
                    <div class="fact-verdict ${isFake ? 'fake' : 'real'}">
                        ${isFake ? 'FAKE' : 'REAL'} NEWS
                    </div>

                    <div class="fact-confidence">
                        Confidence: ${(result.Confidence * 100).toFixed(1)}%
                        <div class="confidence-bar">
                            <div class="confidence-value" 
                                 style="width: ${result.Confidence * 100}%; 
                                        background-color: ${isFake ? '#D22B2B' : '#2B8A3E'}">
                            </div>
                        </div>
                    </div>

                    <div class="model-badges">
                        ${createModelBadge('DistilBERT', result.DistilBERT)}
                        ${createModelBadge('RoBERTa v1', result['RoBERTa v1'])}
                        ${createModelBadge('RoBERTa v2', result['RoBERTa v2'])}
                    </div>

                    ${(result['Latent Features'] || result['Identified Words']) ? `
                        <div class="fact-additional">
                            ${result['Latent Features'] ? `
                                <p><strong>Latent Features:</strong> ${result['Latent Features']}</p>
                            ` : ''}
                            ${result['Identified Words'] ? `
                                <p><strong>Key Words:</strong> ${result['Identified Words']}</p>
                            ` : ''}
                        </div>
                    ` : ''}
                </div>
            `;

        } catch (error) {
            resultContainer.innerHTML = `
                <div class="fact-check-result">
                    <div class="fact-verdict fake">Error analyzing text. Please try again.</div>
                </div>
            `;
        } finally {
            loading.style.display = 'none';
        }
    });
});
