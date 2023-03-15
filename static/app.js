// Get the canvas element and its 2D context
const canvas = document.getElementById('flood-canvas');
const ctx = canvas.getContext('2d');

// Set the canvas size to match the window size
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

// Define the flood color and height
const floodColor = 'blue';
const floodHeight = canvas.height / 2;

// Draw the flood
ctx.fillStyle = floodColor;
ctx.fillRect(0, canvas.height - floodHeight, canvas.width, floodHeight);
