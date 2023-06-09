const weightColor = d3
  .scaleSequentialSqrt(d3.interpolateYlOrRd)
  .domain([0, 1e7]);

// const light = new THREE.AmbietLight(0xfffff,0.5);
// hehe = document.getElementById("globeViz");
// hehe.scene.add(light);

const world = Globe()(document.getElementById("globeViz"))
  .globeImageUrl("//unpkg.com/three-globe/example/img/earth-night.jpg")
  // .globeImageUrl("https://unpkg.com/three-globe@2.25.4/example/img/earth-day.jpg")
  // .globeImageUrl("//flatplanet.sourceforge.net/maps/images/land_ocean_ice_lights_2048.jpg")
  .bumpImageUrl("//unpkg.com/three-globe/example/img/earth-topology.png")
  .backgroundImageUrl("//unpkg.com/three-globe/example/img/night-sky.png")
  .hexBinPointWeight("pop")
  .hexAltitude((d) => d.sumWeight * 6e-8)
  .hexBinResolution(40)
  .hexTopColor((d) => weightColor(d.sumWeight))
  .hexSideColor((d) => weightColor(d.sumWeight))
  .hexBinMerge(true)
  .enablePointerInteraction(false); // performance improvement
fetch("../datasets/world_population.csv")
  .then((res) => res.text())
  .then((csv) =>
    d3.csvParse(csv, ({ lat, lng, pop }) => ({
      lat: +lat,
      lng: +lng,
      pop: +pop,
    }))
  )
  .then((data) => world.hexBinPointsData(data));

// Add auto-rotation
world.controls().autoRotate = true;
world.controls().autoRotateSpeed = 0.8;


// MASTI
// const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

// let interval = null;

// document.querySelector("h1").onmouseover = (event) => {
//   let iteration = 0;

//   clearInterval(interval);

//   interval = setInterval(() => {
//     event.target.innerText = event.target.innerText
//       .split("")
//       .map((letter, index) => {
//         if (index < iteration) {
//           return event.target.dataset.value[index];
//         }

//         return letters[Math.floor(Math.random() * 26)];
//       })
//       .join("");

//     if (iteration >= event.target.dataset.value.length) {
//       clearInterval(interval);
//     }

//     iteration += 1 / 3;
//   }, 60);
// };
