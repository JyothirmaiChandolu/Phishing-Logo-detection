document.getElementById('scan').addEventListener('click', () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.scripting.executeScript(
        {
          target: { tabId: tabs[0].id },
          function: () => {
            const images = [];
            const emailBody = document.querySelector('.a3s');
            const messageContainers = document.querySelectorAll('.adn');
          
            // 1. Sender images
            messageContainers.forEach(container => {
              const senderImage = container.querySelector('img[src^="https://lh3.googleusercontent.com/"]');
              if (senderImage) images.push(senderImage.src);
            });
          
            // 2. Inline body images
            if (emailBody) {
              const bodyImages = Array.from(emailBody.querySelectorAll('img')).map(img => img.src);
              images.push(...bodyImages);
            }
            return images;
          }
        }, async (injectionResults) => {
          const imageUrls = injectionResults[0].result;
          const container = document.getElementById('resultContainer');
          container.innerHTML = "";
          for (const url of imageUrls) {
            if (!url.startsWith('http')) {
              console.warn("Skipping invalid URL:", url);
              continue;
            }
  
            const resultDiv = document.createElement('div');
            resultDiv.textContent = `Scanning ${url}...`;
            container.appendChild(resultDiv);
  
            try {
              const response = await fetch(url);
              const blob = await response.blob();
              const formData = new FormData();
              formData.append('image', blob, 'image.jpg');
  
              const apiResponse = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                body: formData
              });
  
              const data = await apiResponse.json();
                const isFake = knoim.includes(url);
                const label = isFake
                  ? '❌ Fake'
                  : (data.result === 'fake' ? '❌ Fake' : '✅ Genuine');
    
                resultDiv.innerHTML = `
                  <div style="margin: 10px; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
                    <img src="${url}" width="100" style="margin-bottom: 10px;"><br>
                    <strong>Result: ${label}</strong><br>
                  </div>
                `;
            } catch (error) {
              console.error("Failed to process image:", error);
              resultDiv.innerHTML = `Error processing ${url}`;
            }
          }
        }
      );
    });
  });
  
  
