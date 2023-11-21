// Source of webcam related code:
// https://developer.mozilla.org/en-US/docs/Web/API/MediaStream_Recording_API/Recording_a_media_element.

// Source of MediaPipe related code:
// https://github.com/google/mediapipe/blob/master/docs/getting_started/javascript.md

const videoElement = document.getElementById('preview');
var clipLandmarks = [];
// The camera is already recording. This will be set to true when we want to
// actually save the landmarks.
var saveLandmarks = false;

function onResults(results) {
  // Collect the keypoints and add them to the array.
  // Only when we are actively recording.
  if (!saveLandmarks) {
    return;
  }

  var frameLandmarks = [];
  if (results.poseLandmarks) {
    for (var i = 0; i < results.poseLandmarks.length; i++) {
        frameLandmarks.push(results.poseLandmarks[i].x);
        frameLandmarks.push(results.poseLandmarks[i].y);
        frameLandmarks.push(results.poseLandmarks[i].z);
    }
  } else {
    for (var i = 0; i < 33; i++) {
        frameLandmarks.push(NaN);
        frameLandmarks.push(NaN);
        frameLandmarks.push(NaN);
    }
  }

  if (results.leftHandLandmarks) {
    for (var i = 0; i < results.leftHandLandmarks.length; i++) {
        frameLandmarks.push(results.leftHandLandmarks[i].x);
        frameLandmarks.push(results.leftHandLandmarks[i].y);
        frameLandmarks.push(results.leftHandLandmarks[i].z);
    }
  } else {
    for (var i = 0; i < 21; i++) {
        frameLandmarks.push(NaN);
        frameLandmarks.push(NaN);
        frameLandmarks.push(NaN);
    }
  }

  if (results.rightHandLandmarks) {
    for (var i = 0; i < results.rightHandLandmarks.length; i++) {
        frameLandmarks.push(results.rightHandLandmarks[i].x);
        frameLandmarks.push(results.rightHandLandmarks[i].y);
        frameLandmarks.push(results.rightHandLandmarks[i].z);
    }
  } else {
    for (var i = 0; i < 21; i++) {
        frameLandmarks.push(NaN);
        frameLandmarks.push(NaN);
        frameLandmarks.push(NaN);
    }
  }

  if (results.faceLandmarks) {
    for (var i = 0; i < results.faceLandmarks.length; i++) {
        frameLandmarks.push(results.faceLandmarks[i].x);
        frameLandmarks.push(results.faceLandmarks[i].y);
        frameLandmarks.push(results.faceLandmarks[i].z);
    }
  } else {
    for (var i = 0; i < 468; i++) {
        frameLandmarks.push(NaN);
        frameLandmarks.push(NaN);
        frameLandmarks.push(NaN);
    }
  }

  clipLandmarks.push(frameLandmarks);
}

let startButton = document.getElementById("startButton");
startButton.disabled = true;

const holistic = new Holistic({locateFile: (file) => {
  return `mediapipe/${file}`;
}});
holistic.setOptions({
  modelComplexity: 2,
  smoothLandmarks: true,
  enableSegmentation: false,
  smoothSegmentation: false,
  refineFaceLandmarks: false,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});
holistic.onResults(onResults);
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await holistic.send({image: videoElement});
  },
  width: 320,
  height: 240
});
camera.start();

startButton.disabled = false;  // Enable button once MediaPipe is ready.

// Time between clicking "record" and starting the recording.
let waitTimeMS = 1500;
// Duration of the recording.
let recordingTimeMS = 4000;

let resultsDiv = document.getElementById("results");
let continueButton = document.getElementById("continueButton");
var spinner = document.getElementById("spinner");
var exampleVideo = document.getElementById("example");
var exampleGlossName = document.getElementById("exampleGlossName");
let countdownDisplay = document.getElementById("countdown");

// Dutch translation of the ID Glosses that we record.
let glosses = [
    "Bouwen",
    "Waarom",
    "Hebben",
    "Paard",
    "Melk",
    "Herfst",
    "Valentijn",
    "Telefoneren",
    "Straat",
    "Haas",
    "Hond",
    "Rusten",
    "School",
    "Onthouden",
    "Wat",
    "Vliegtuig",
    "Bel",
    "Kat",
    "Mama",
    "Papa"
];

// ID Glosses that we record. Order should be the same as `glosses`.
let id_glosses = [
    "BOUWEN-G-1906",
    "WAAROM-A-13564",
    "HEBBEN-A-4801",
    "PAARD-A-8880",
    "MELK-B-7418",
    "HERFST-B-4897",
    "VALENTIJN-A-16235",
    "TELEFONEREN-D-11870",
    "STRAAT-A-11560",
    "HAAS-A-16146",
    "HOND-A-5052",
    "RUSTEN-B-10250",
    "SCHOOL-A-10547",
    "ONTHOUDEN-A-8420",
    "WAT-A-13657",
    "VLIEGTUIG-B-13187",
    "KLEPELBEL-A-1166",
    "POES-G-9372",
    "MOEDER-A-7676",
    "VADER-G-8975"
];

/**
 * Set the example video for the current gloss.
 */
function updateGloss() {
    let currentGloss = glosses[currentGlossIndex];
    let id_gloss = id_glosses[currentGlossIndex];

    let prefix = id_gloss.slice(0, 2);
    let url = 'https://vlaamsegebarentaal.be/signbank/dictionary/protected_media/glossvideo/' + prefix + '/' + id_gloss + '.mp4';

    exampleGlossName.innerHTML = currentGloss;
    exampleVideo.src = url;
}

// On page load, set the first gloss.
var currentGlossIndex = random(0, glosses.length);
updateGloss();

/**
 * Remove the spinner when the search operation is done.
 */
function done() {
    spinner.classList.add("visually-hidden");
}

/**
 * Get a random integer between `mn` and `mx` (exclusive).
 */
function random(mn, mx) {
    return Math.floor(Math.random() * (mx - mn) + mn);
}

/**
 * Reset the page when the "continue" button is clicked.
 */
continueButton.addEventListener("click", () => {
    for (var i = 0; i < 3; i++) {
        for (var j = 0; j < 3; j++) {
            var div = document.getElementById('row' + i + 'col' + j);
            while(div.firstChild){
                div.removeChild(div.firstChild);
            }
        }
    }

    startButton.disabled = false;

    clipLandmarks = [];
    currentGlossIndex = random(0, glosses.length);
    console.log(currentGlossIndex);
    updateGloss();
}, false);


/**
 * Wait for an amount of milliseconds.
 */
function wait(delayInMS) {
  return new Promise((resolve) => setTimeout(resolve, delayInMS));
}

/**
 * Make a recording of a certain duration from the webcam.
 */
function startRecording(lengthInMS) {
  clipLandmarks = [];  // Clear previous landmarks, if any.
  console.log('Starting recording...');
  saveLandmarks = true;
  videoElement.classList.add('active');
  let recorded = wait(lengthInMS).then(
    () => {
        saveLandmarks = false;
        videoElement.classList.remove('active');
        console.log('Stopping recording...');

        console.log('Starting search...');
        search();
        console.log('Search completed...')
    },
  );
}

startButton.addEventListener("click", () => {
    startButton.disabled = true;
    // Show countdown.
    var timeLeft = waitTimeMS;
    countdownDisplay.style.display = 'block';
    var downloadTimer = setInterval(function(){
        if(timeLeft <= 0) {
            clearInterval(downloadTimer);
        }
        countdownDisplay.innerHTML = (timeLeft / 1000).toFixed(1);
        timeLeft -= 100;
    }, 100);

    wait(waitTimeMS).then(() => {
        // Stop showing countdown.
        countdownDisplay.style.display = 'none';
        // Record video with keypoints.
        startRecording(recordingTimeMS);
    });
}, false);

function search() {
    spinner.classList.remove("visually-hidden");

    // Send the keypoint data to the server and get a response back...
    fetch('http://localhost:5000/search?' + new URLSearchParams({
            gtgloss: id_glosses[currentGlossIndex],
        }), { method: "POST", body: clipLandmarks, headers: {'CONSENT': true} })
        .then(response => {
            if (response.ok) {
                return response;
            }
            else {
                throw Error(`Server returned ${response.status}: ${response.statusText}`);
            }
        })
        .then(response => response.json())
        .then(data => setResults(data.results));
}

function setResults(results) {
    spinner.classList.add("visually-hidden");
    results.forEach((result, index) => {
        let prefix = result.slice(0, 2);
        let url = 'https://vlaamsegebarentaal.be/signbank/dictionary/protected_media/glossvideo/' + prefix + '/' + result + '.mp4';

        let rawGloss = result.slice(0, result.indexOf("-"));
        let numericId = result.slice(result.lastIndexOf("-") + 1);
        rawGloss = rawGloss.charAt(0) + rawGloss.substring(1).toLowerCase();

        let row_index = index % 3;  // 3 rows.
        let col_index = Math.floor(index / 3);

        var resultDiv = document.getElementById("row" + row_index + "col" + col_index);
        var resultGloss = document.createElement("h5");
        resultGloss.innerHTML = rawGloss;
        resultGloss.classList.add("card-title");
        var resultLink = document.createElement("a");
        resultLink.href = "https://woordenboek.vlaamsegebarentaal.be/gloss/" + rawGloss + "?sid=" + numericId;
        resultLink.innerHTML = "Details";

        var videoEl = document.createElement("video");
        videoEl.classList.add("center");
        videoEl.width = 320;
        videoEl.height = 240;
        videoEl.controls = "true";
        videoEl.src = url;

        var br = document.createElement("br");

        resultDiv.children = [];
        resultDiv.appendChild(resultGloss);
        resultDiv.appendChild(videoEl);
        resultDiv.appendChild(br);
        resultDiv.appendChild(resultLink);
        resultDiv.appendChild(document.createElement("br"));
    });
}
