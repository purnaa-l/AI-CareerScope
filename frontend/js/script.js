const facts = [
  "💡 Did you know? Data Scientist was the most in-demand AI role in 2024!",
  "💡 Did you know? AI job postings in India increased by 46% in just one year!",
  "💡 Did you know? The average salary of an AI Engineer in the US crossed $130,000 in 2025!",
  "💡 Did you know? Over 80% of companies are using AI to automate tasks and decision-making!",
  "💡 Did you know? Germany, India, and the US are the top countries hiring AI professionals!"
];

let factIndex = 0;
let rotationInterval = null;

const factBox = document.getElementById("factBox");
const factText = document.getElementById("factText");

function rotateFact() {
  factIndex = (factIndex + 1) % facts.length;
  factText.textContent = facts[factIndex];
}

window.addEventListener("scroll", function () {
  const scrollY = window.scrollY;
  const docHeight = document.documentElement.scrollHeight - window.innerHeight;
  const scrollRatio = scrollY / docHeight;

  if (scrollRatio >= 0.2 && scrollRatio <= 0.9) {
    if (factBox.style.display !== "block") {
      factBox.style.display = "block";
      rotateFact(); // change to next fact immediately
      rotationInterval = setInterval(rotateFact, 6000);
    }
  } else {
    if (factBox.style.display === "block") {
      factBox.style.display = "none";
      clearInterval(rotationInterval);
      rotationInterval = null;
    }
  }
});


//Quotes
const quotes = [
  {
    text: "“Success in the 21st century is determined by your ability to adapt to AI.”",
    author: "– Sundar Pichai"
  },
  {
    text: "“AI will be the best or worst thing ever for humanity. We need to get it right.”",
    author: "– Elon Musk"
  },
  {
    text: "“In 10 years, over 50% of jobs will require some knowledge of AI.”",
    author: "– Andrew Ng"
  },
  {
    text: "“The secret to career success today? Learn to work with AI, not against it.”",
    author: "– Fei-Fei Li"
  },
  
  {
    text: "“The jobs of the future are those that work in harmony with intelligent systems.”",
    author: "– Satya Nadella"
  }
];

let quoteIndex = 0;
const quoteDisplay = document.getElementById('quoteDisplay');
const prevQuote = document.getElementById('prevQuote');
const nextQuote = document.getElementById('nextQuote');


function updateQuote() {
  // Add fade-in class
  quoteDisplay.classList.add("fade-in");

  // Change content
  quoteDisplay.innerHTML = `
    <p class="quote">${quotes[quoteIndex].text}</p>
    <span class="author">${quotes[quoteIndex].author}</span>
  `;

  // Remove fade-in after animation ends
  setTimeout(() => {
    quoteDisplay.classList.remove("fade-in");
  }, 500); // match duration in CSS
}


prevQuote.addEventListener("click", () => {
  quoteIndex = (quoteIndex - 1 + quotes.length) % quotes.length;
  updateQuote();
});

nextQuote.addEventListener("click", () => {
  quoteIndex = (quoteIndex + 1) % quotes.length;
  updateQuote();
});

// Initial load
updateQuote();
