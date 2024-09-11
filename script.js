document.getElementById('videoUpload').addEventListener('change', function(event) {
    const file = event.target.files[0];
    const videoPreview = document.getElementById('videoPreview');
    const videoDetails = document.getElementById('videoDetails');
    const spinner = document.getElementById('spinner');

    // Clear previous content
    videoPreview.innerHTML = '';
    videoDetails.innerHTML = '';

    if (file) {
        // Check if the file is a video
        if (!file.type.startsWith('video/')) {
            alert('Unsupported file format. Please upload a video file.');
            return;
        }

        // Show loading spinner
        spinner.style.display = 'block';

        // Simulate video processing delay
        setTimeout(() => {
            // Hide loading spinner
            spinner.style.display = 'none';

            // Show video details
            const details = `
                <p><strong>File Name:</strong> ${file.name}</p>
                <p><strong>File Size:</strong> ${(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                <p><strong>File Type:</strong> ${file.type}</p>
            `;
            videoDetails.innerHTML = details;

            // Show video preview
            const videoElement = document.createElement('video');
            videoElement.src = URL.createObjectURL(file);
            videoElement.controls = true;
            videoPreview.appendChild(videoElement);
        }, 2000); // Simulate a 2-second delay for processing
    }
});
