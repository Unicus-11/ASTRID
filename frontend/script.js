document.addEventListener('DOMContentLoaded', () => {
   const barcodeContainer = document.getElementById('barcode-container');
   const interactiveArea = document.getElementById('interactive-area');
   
   const numBars = 110; 
   const bars = [];
   
   const BASE_HEIGHT = 8;
   const MAX_HEIGHT = 65; 
   
   const COLOR_SILVER = '#4a4a4a';
   const COLOR_GLOW = 'rgba(111, 255, 126, 0.5)';
   const COLOR_NEON = '#6fff7e';

   let mouseX = window.innerWidth / 2; // Default center position so it moves right away
   let isHovering = true; // Always active motion state

   for (let i = 0; i < numBars; i++) {
       const bar = document.createElement('div');
       bar.classList.add('barcode-bar');
       barcodeContainer.appendChild(bar);
       bars.push({
           el: bar,
           phaseOffset: i * 0.15
       });
   }

   interactiveArea.addEventListener('mousemove', (e) => {
       mouseX = e.clientX;
       isHovering = true;
   });

   interactiveArea.addEventListener('mouseleave', () => {
       // Keeps moving smoothly with an automatic center position when the mouse leaves
       mouseX = window.innerWidth / 2;
   });

   function interpolateColor(factor) {
       if (factor > 0.7) return COLOR_NEON;
       if (factor > 0.25) return COLOR_GLOW;
       return COLOR_SILVER;
   }

   function animate() {
       const time = Date.now() * 0.003;

       bars.forEach((barData, i) => {
           const rect = barData.el.getBoundingClientRect();
           const barCenterX = rect.left + (rect.width / 2);
           
           const distance = Math.abs(mouseX - barCenterX);
           const spread = 120; 

           // Cursor proximity intensity
           const cursorIntensity = Math.exp(-Math.pow(distance, 2) / (2 * Math.pow(spread, 2)));

           // Continuous background traveling wave motion so it's always active
           const continuousWave = (Math.sin(time + barData.phaseOffset) + 1) * 0.35;

           // Combine both cursor interaction and continuous baseline movement
           const combinedIntensity = Math.min(1, cursorIntensity + continuousWave);

           const targetHeight = BASE_HEIGHT + ((MAX_HEIGHT - BASE_HEIGHT) * combinedIntensity);
           const targetColor = interpolateColor(combinedIntensity);

           barData.el.style.height = `${targetHeight}px`;
           barData.el.style.backgroundColor = targetColor;
       });

       requestAnimationFrame(animate);
   }

   animate();
});



document.addEventListener("DOMContentLoaded", function () {
   const observerOptions = {
       root: null,
       rootMargin: '0px',
       threshold: 0.15 // Triggers when 15% of the element is visible
   };

   const observer = new IntersectionObserver((entries, observer) => {
       entries.forEach(entry => {
           if (entry.isIntersecting) {
               entry.target.classList.add('is-visible');
               // Optional: stop observing once animated
               observer.unobserve(entry.target);
           }
       });
   }, observerOptions);

   // Target the section header and all showcase cards
   const animatedElements = document.querySelectorAll('.analytics-showcase-section .section-header, .analytics-showcase-section .showcase-card');
   
   animatedElements.forEach(el => {
       observer.observe(el);
   });
});



document.querySelectorAll(".faq-question").forEach(button => {

    button.addEventListener("click", () => {

        const item = button.closest(".faq-item");
        const isActive = item.classList.contains("active");

        document.querySelectorAll(".faq-item").forEach(otherItem => {
            otherItem.classList.remove("active");
        });

        if (!isActive) {
            item.classList.add("active");
        }

    });

});
