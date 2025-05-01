/*AIzaSyBqXPdBFZyHCEQNOb8M7ZHkFP-JvnyU9yA
https://safebrowsing.googleapis.com/v4/threatMatches:find?key=${apiKey}*/
// content.js
console.log('Content script loaded'); // Debug log
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'scanURLs') {
    console.log('Content script received scanURLs request');
    scanURLs().then(results => {
      console.log('Sending results:', results);
      sendResponse({ results });
    }).catch(error => {
      console.error('Error in scanURLs:', error);
      sendResponse({ results: [{ url: 'Error', status: 'Failed to scan' }] });
    });
    return true; // Indicates async response
  }
});

function scanURLs() {
  console.log('scanURLs function called');
  const API_KEY = 'AIzaSyBqXPdBFZyHCEQNOb8M7ZHkFP-JvnyU9yA'; // Ensure this is valid
  const API_URL = `https://safebrowsing.googleapis.com/v4/threatMatches:find?key=${API_KEY}`; // Use backticks for template literal
  
  const links = Array.from(document.getElementsByTagName('a'))
    .map(a => a.href)
    .filter(href => href.startsWith('http'));
  
  console.log('Found links:', links);
  if (links.length === 0) {
    return Promise.resolve([{ url: 'No URLs found', status: 'N/A' }]);
  }

  const requestBody = {
    client: {
      clientId: 'url-safety-scanner',
      clientVersion: '1.0'
    },
    threatInfo: {
      threatTypes: ['MALWARE', 'SOCIAL_ENGINEERING', 'UNWANTED_SOFTWARE', 'POTENTIALLY_HARMFUL_APPLICATION'],
      platformTypes: ['ANY_PLATFORM'],
      threatEntryTypes: ['URL'],
      threatEntries: links.map(url => ({ url }))
    }
  };

  console.log('Sending API request');
  return fetch(API_URL, {
    method: 'POST',
    body: JSON.stringify(requestBody),
    headers: { 'Content-Type': 'application/json' }
  })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      console.log('API response:', data);
      const results = links.map(url => {
        const isUnsafe = data.matches && data.matches.some(match => url.includes(match.threat.url));
        return { url, status: isUnsafe ? 'Unsafe' : 'Safe' };
      });
      return results;
    })
    .catch(error => {
      console.error('Error checking URLs:', error.message);
      return links.map(url => ({ url, status: 'Error' }));
    });
}
