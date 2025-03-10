document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const snap = document.getElementById('snap');
    const image = document.getElementById('image');
    const longitude = document.getElementById('longitude');
    const latitude = document.getElementById('latitude');

    // Access camera
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            video.srcObject = stream;
        })
        .catch(function(err) {
            console.error("Error accessing camera: ", err);
        });

    // Capture snapshot
    snap.addEventListener('click', function() {
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, 300, 200);
        image.value = canvas.toDataURL('image/png');
    });

    // Get geolocation
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(function(position) {
            longitude.value = position.coords.longitude;
            latitude.value = position.coords.latitude;
        }, function(error) {
            console.error("Error getting geolocation: ", error);
        });
    } else {
        console.error("Geolocation is not supported by this browser.");
    }
});