document.getElementById('scanButton').addEventListener('click', () => {
  console.log('Scan button clicked!');
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (!tabs || tabs.length === 0) {
      console.error('No active tab found');
      document.getElementById('results').innerHTML = '<p>Error: No active tab found.</p>';
      return;
    }

    const tabId = tabs[0].id;

    //Inject content.js if not already
    chrome.scripting.executeScript(
      {
        target: { tabId },
        files: ['content.js']
      },
      () => {
        if (chrome.runtime.lastError) {
          console.error('Injection failed:', chrome.runtime.lastError.message);
          document.getElementById('results').innerHTML = `<p>Error: ${chrome.runtime.lastError.message}</p>`;
          return;
        }

        //Send message to the now-injected script
        chrome.tabs.sendMessage(tabId, { action: 'scanURLs' }, (response) => {
          console.log('Response from content script:', response);
          if (chrome.runtime.lastError) {
            console.error('Runtime error:', chrome.runtime.lastError.message);
            document.getElementById('results').innerHTML = '<p>Error: Could not connect to content script.</p>';
            return;
          }

          const resultDiv = document.getElementById('results');
          if (response && response.results) {
            resultDiv.innerHTML = '<h2>Scan Results:</h2>';
            response.results.forEach(({ url, status }) => {
              let statusClass = '';
              if (status === 'Safe') statusClass = 'safe';
              else if (status === 'Unsafe') statusClass = 'unsafe';
              else statusClass = 'error';

              resultDiv.innerHTML += `<p>${url}: <strong class="${statusClass}">${status}</strong></p>`;
            });
          } else {
            resultDiv.innerHTML = '<p>Error scanning URLs or no URLs found.</p>';
          }
        });
      }
    );
  });
});
