let images = document.querySelectorAll('img');
let formData = new FormData();
if (images.length > 0) {
  let imgSrc = images[0].src;
  fetch(imgSrc)
    .then(response => response.blob())
    .then(blob => {
      formData.append("image", blob, "image.jpg");
      chrome.runtime.sendMessage({ action: "classifyImage", formData: formData }, function(response) {
        console.log("Classification Result:", response);
        alert('This image is ' + response.classification);
      });
    });
}
