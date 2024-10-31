chrome.action.onClicked.addListener(async (tab) => {
    if (tab.url.includes("twitter.com") || tab.url.includes("x.com")) {
        try {
            // First, inject the script file
            await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                files: ['inject.js']
            });
            
            // Then execute the function
            await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                function: () => {
                    window.injectFactChecker();
                }
            });
        } catch (err) {
            console.error('Failed to inject script:', err);
        }
    }
});
