const micBtn = document.getElementById('mic-btn');
const output = document.getElementById('output');

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = new SpeechRecognition();

recognition.lang = 'en-US';
recognition.interimResults = true;

micBtn.addEventListener('click', () => {
  recognition.start();
  output.textContent = 'Listening... 🎙️';
});

recognition.onresult = (event) => {
  const transcript = Array.from(event.results)
    .map(result => result[0])
    .map(result => result.transcript)
    .join('');

  output.textContent = transcript;

  if (event.results[0].isFinal) {
    output.textContent = `You said: "${transcript}"`;
    // future: trigger voice actions
  }
};

recognition.onerror = (e) => {
  output.textContent = `Error: ${e.error}`;
};
