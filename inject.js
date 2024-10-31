window.injectFactChecker = function() {
    // Add our styles
    const style = document.createElement('style');
    style.textContent = `
        .fact-check-container {
            padding: 4px 0;
            border-top: 1px solid rgb(239, 243, 244);
            margin-top: 4px;
        }

        .fact-check-button {
            background-color: #D22B2B;
            color: white;
            border: none;
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 13px;
            cursor: pointer;
            margin: 4px 0;
            display: inline-flex;
            align-items: center;
            transition: background-color 0.2s;
        }

        .fact-check-button:hover {
            background-color: #B71C1C;
        }

        .fact-check-button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }

        .fact-check-result {
            margin: 8px 0;
            padding: 12px;
            border-radius: 8px;
            font-size: 14px;
            background-color: rgba(43, 45, 66, 0.05);
            border-left: 4px solid #D22B2B;
        }

        .fact-check-result.real {
            border-left-color: #2B8A3E;
        }

        .fact-verdict {
            font-weight: bold;
            margin-bottom: 8px;
        }

        .fact-verdict.fake {
            color: #D22B2B;
        }

        .fact-verdict.real {
            color: #2B8A3E;
        }

        .fact-model-details {
            font-size: 12px;
            color: #666;
            margin: 8px 0;
        }

        .fact-confidence {
            margin: 8px 0;
        }

        .confidence-bar {
            height: 4px;
            background: #eee;
            border-radius: 2px;
            overflow: hidden;
        }

        .confidence-value {
            height: 100%;
            transition: width 0.3s ease;
        }

        .fact-additional {
            font-size: 12px;
            color: #666;
            margin-top: 8px;
            border-top: 1px solid #eee;
            padding-top: 8px;
        }

        .model-badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            margin-right: 4px;
            font-size: 11px;
            color: white;
        }

        .model-badge.fake {
            background-color: rgba(210, 43, 43, 0.8);
        }

        .model-badge.real {
            background-color: rgba(43, 138, 62, 0.8);
        }

        .analyzing-spinner {
            display: inline-block;
            width: 12px;
            height: 12px;
            border: 2px solid #fff;
            border-top-color: transparent;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 6px;
            vertical-align: middle;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    `;
    document.head.appendChild(style);

    async function analyzeText(text) {
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
            return data[0];
        } catch (error) {
            console.error('Error:', error);
            throw error;
        }
    }

    function createModelBadge(modelName, isFake) {
        return `
            <span class="model-badge ${isFake ? 'fake' : 'real'}">
                ${modelName}: ${isFake ? 'FAKE' : 'REAL'}
            </span>
        `;
    }

    function addButtonToTweet(tweet) {
        // Check if we've already added a button
        if (tweet.querySelector('.fact-check-container')) {
            return;
        }

        const textElement = tweet.querySelector('[data-testid="tweetText"]');
        if (!textElement) {
            return;
        }

        // Create container for fact check elements
        const container = document.createElement('div');
        container.className = 'fact-check-container';

        const button = document.createElement('button');
        button.className = 'fact-check-button';
        button.textContent = 'Check Facts';
        container.appendChild(button);
        
        button.addEventListener('click', async () => {
            const tweetText = textElement.textContent.trim();
            if (!tweetText) {
                return;
            }

            try {
                // Update button to loading state
                button.disabled = true;
                button.innerHTML = '<span class="analyzing-spinner"></span>Analyzing...';

                // Remove any existing result
                const existingResult = container.querySelector('.fact-check-result');
                if (existingResult) {
                    existingResult.remove();
                }

                const result = await analyzeText(tweetText);
                
                const resultElement = document.createElement('div');
                resultElement.className = `fact-check-result ${result.Ensemble ? 'fake' : 'real'}`;
                
                // Main verdict
                const verdict = document.createElement('div');
                verdict.className = `fact-verdict ${result.Ensemble ? 'fake' : 'real'}`;
                verdict.textContent = `This post appears to be ${result.Ensemble ? 'FAKE' : 'REAL'} NEWS`;
                resultElement.appendChild(verdict);

                // Confidence bar
                const confidenceSection = document.createElement('div');
                confidenceSection.className = 'fact-confidence';
                confidenceSection.innerHTML = `
                    Confidence: ${(result.Confidence * 100).toFixed(1)}%
                    <div class="confidence-bar">
                        <div class="confidence-value" 
                             style="width: ${result.Confidence * 100}%; 
                                    background-color: ${result.Ensemble ? '#D22B2B' : '#2B8A3E'}">
                        </div>
                    </div>
                `;
                resultElement.appendChild(confidenceSection);

                // Model details
                const modelDetails = document.createElement('div');
                modelDetails.className = 'fact-model-details';
                modelDetails.innerHTML = `
                    ${createModelBadge('DistilBERT', result.DistilBERT)}
                    ${createModelBadge('RoBERTa v1', result['RoBERTa v1'])}
                    ${createModelBadge('RoBERTa v2', result['RoBERTa v2'])}
                `;
                resultElement.appendChild(modelDetails);

                // Additional information
                if (result['Latent Features'] || result['Identified Words']) {
                    const additional = document.createElement('div');
                    additional.className = 'fact-additional';
                    additional.innerHTML = `
                        ${result['Latent Features'] ? `<div><strong>Latent Features:</strong> ${result['Latent Features']}</div>` : ''}
                        ${result['Identified Words'] ? `<div><strong>Key Words:</strong> ${result['Identified Words']}</div>` : ''}
                    `;
                    resultElement.appendChild(additional);
                }

                container.appendChild(resultElement);

            } catch (error) {
                const errorElement = document.createElement('div');
                errorElement.className = 'fact-check-result';
                errorElement.textContent = 'Error analyzing post. Please try again.';
                container.appendChild(errorElement);
            } finally {
                // Reset button state
                button.disabled = false;
                button.textContent = 'Check Facts';
            }
        });

        // Find where to insert the container
        const actionBar = tweet.querySelector('[role="group"]');
        if (actionBar) {
            // Insert after the action bar
            actionBar.parentNode.insertBefore(container, actionBar.nextSibling);
        }
    }

    // Process existing tweets
    const tweets = document.querySelectorAll('article[data-testid="tweet"]');
    tweets.forEach(addButtonToTweet);

    // Create observer for new tweets
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            mutation.addedNodes.forEach((node) => {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    const tweets = node.querySelectorAll('article[data-testid="tweet"]');
                    tweets.forEach(addButtonToTweet);
                }
            });
        });
    });

    // Start observing the timeline
    const timeline = document.querySelector('main');
    if (timeline) {
        observer.observe(timeline, {
            childList: true,
            subtree: true
        });
    }

    console.log('Fact checker initialized - buttons added to tweets');
}