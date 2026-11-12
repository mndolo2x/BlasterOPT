/* Illustrative terrain and observations. All geometry and readings are fictional. */
(() => {
  "use strict";
  const viewport = document.getElementById("viewport");
  const loading = document.getElementById("loading");
  if (!window.THREE || !THREE.OrbitControls) {
    loading.textContent = "The 3D viewer could not load. Refresh after updating the app files.";
    return;
  }
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({antialias: true, alpha: true, preserveDrawingBuffer: true});
  } catch (error) {
    loading.textContent = "3D rendering is unavailable. Enable graphics acceleration in your browser and reload.";
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  viewport.prepend(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.5, 1400);
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 65;
  controls.maxDistance = 520;
  controls.maxPolarAngle = Math.PI * 0.48;
  const world = new THREE.Group();
  scene.add(world);
  scene.add(new THREE.HemisphereLight(0xd4ebff, 0x575345, 0.85));
  const sun = new THREE.DirectionalLight(0xffe1b3, 2.2);
  sun.position.set(-90, 170, 100);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, {left: -160, right: 160, top: 140, bottom: -140, far: 500});
  sun.shadow.bias = -0.00035;
  sun.shadow.normalBias = 0.3;
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0x93b9d1, 0.45);
  fill.position.set(80, 50, -80);
  scene.add(fill);

  const material = (color, extra = {}) => new THREE.MeshStandardMaterial({color, roughness: 0.94, ...extra});
  const grid = new THREE.GridHelper(600, 60, 0x48616e, 0x283d49);
  grid.position.y = -4;
  grid.material.transparent = true;
  grid.material.opacity = 0.28;
  scene.add(grid);

  function surface(x, z) {
    const u = z + Math.sin(x * 0.026) * 3.6 + Math.sin(x * 0.071) * 0.75;
    if (u > 32) return 3;
    if (u > 19) return 3 + (32 - u);
    if (u > 0) return 16;
    if (u > -13) return 16 - u;
    if (u > -38) return 29;
    if (u > -51) return 29 + (-38 - u);
    return 42;
  }
  const grain = (x, y, z) => {
    const value = Math.sin(x * 12.9898 + y * 17.111 + z * 78.233) * 43758.5453;
    return value - Math.floor(value);
  };
  function rockColor(x, y, z, geological) {
    const ore = ((x - 20) / 34) ** 2 + ((z + 15) / 67) ** 2 < 1 && y < 31;
    const base = !geological ? 0xb19573 : ore ? 0x648c8a :
      y > 30 ? 0xc6a275 : y > 17 ? 0xa78864 : y > 6 ? 0x987656 : 0x715d4d;
    return new THREE.Color(base).multiplyScalar(0.85 + grain(x, y, z) * 0.27);
  }
  const vertices = [];
  function triangle(a, b, c) { vertices.push(...a, ...b, ...c); }
  const step = 2;
  for (let x = -100; x < 100; x += step) {
    for (let z = -70; z < 70; z += step) {
      const a = [x, surface(x, z), z], b = [x + step, surface(x + step, z), z];
      const c = [x, surface(x, z + step), z + step], d = [x + step, surface(x + step, z + step), z + step];
      triangle(a, c, b); triangle(b, c, d);
    }
  }
  function side(x1, z1, x2, z2) {
    const top1 = surface(x1, z1), top2 = surface(x2, z2);
    for (let y = -3; y < Math.max(top1, top2); y += 2) {
      const a = [x1, Math.min(y, top1), z1], b = [x2, Math.min(y, top2), z2];
      const c = [x1, Math.min(y + 2, top1), z1], d = [x2, Math.min(y + 2, top2), z2];
      triangle(a, b, c); triangle(b, d, c);
    }
  }
  for (let x = -100; x < 100; x += step) { side(x, -70, x + step, -70); side(x, 70, x + step, 70); }
  for (let z = -70; z < 70; z += step) { side(-100, z, -100, z + step); side(100, z, 100, z + step); }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  geometry.computeVertexNormals();
  function setGeology(enabled) {
    const colors = [];
    for (let index = 0; index < vertices.length; index += 3) {
      const color = rockColor(vertices[index], vertices[index + 1], vertices[index + 2], enabled);
      colors.push(color.r, color.g, color.b);
    }
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  }
  setGeology(true);
  const terrain = new THREE.Mesh(geometry, material(0xffffff, {vertexColors: true, side: THREE.DoubleSide, flatShading: true}));
  terrain.castShadow = true;
  terrain.receiveShadow = true;
  world.add(terrain);
  const wire = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({color: 0xb8d9d8, wireframe: true, transparent: true, opacity: 0.13}));
  wire.position.y = 0.07;
  wire.visible = false;
  world.add(wire);

  function box(parent, size, position, mat) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), mat);
    mesh.position.set(...position);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    parent.add(mesh);
    return mesh;
  }
  // Low foundation plinth grounds the model and exposes the outer geological faces.
  box(world, [200, 1, 140], [0, -3.5, 0], material(0x3e4749));
  box(world, [181, 0.13, 10], [-3, 3.11, 53], material(0x877969));
  const stripeMat = material(0xd0b68a);
  for (let x = -88; x < 84; x += 12) box(world, [4, 0.04, 0.25], [x, 3.20, 53], stripeMat);
  const rockMat = material(0x9c856a);
  const rockGeometry = new THREE.DodecahedronGeometry(1, 0);
  for (let index = 0; index < 210; index++) {
    const x = -98 + grain(index, 1, 2) * 196;
    const z = -68 + grain(index, 2, 3) * 136;
    if (z > 45) continue;
    const rock = new THREE.Mesh(rockGeometry, rockMat);
    const scale = 0.22 + grain(index, 3, 4) * 0.62;
    rock.scale.set(scale * 1.5, scale * 0.65, scale);
    rock.position.set(x, surface(x, z) + scale * 0.2, z);
    rock.rotation.set(index, index * 0.7, index * 0.3);
    rock.castShadow = true;
    world.add(rock);
  }

  const equipment = new THREE.Group();
  world.add(equipment);
  const yellow = material(0xe6b343), dark = material(0x222c30), glass = material(0x355e6b, {metalness: 0.3, roughness: 0.32});
  function wheel(parent, x, y, z, radius = 1.5) {
    const tire = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, 1.05, 16), dark);
    tire.rotation.x = Math.PI / 2;
    tire.position.set(x, y, z);
    tire.castShadow = true;
    parent.add(tire);
    const hub = new THREE.Mesh(new THREE.CylinderGeometry(radius * 0.42, radius * 0.42, 1.12, 12), yellow);
    hub.rotation.x = Math.PI / 2;
    hub.position.copy(tire.position);
    parent.add(hub);
  }
  const truck = new THREE.Group();
  equipment.add(truck);
  box(truck, [10, 1, 4.4], [0, 2, 0], dark);
  box(truck, [3, 2.8, 4.3], [3.8, 3.7, 0], yellow);
  box(truck, [1.9, 1.4, 3.85], [4.4, 4.5, 0], glass);
  box(truck, [7, 0.6, 5.4], [-1.7, 3.3, 0], yellow);
  box(truck, [7, 2.3, 0.35], [-1.7, 4.4, -2.55], yellow);
  box(truck, [7, 2.3, 0.35], [-1.7, 4.4, 2.55], yellow);
  box(truck, [0.35, 2.3, 5], [-5, 4.4, 0], yellow);
  box(truck, [5.8, 1.3, 4.4], [-1.6, 4, 0], rockMat);
  for (const x of [-3.8, -0.8, 3.9]) for (const z of [-2.25, 2.25]) wheel(truck, x, 1.5, z);
  truck.position.set(-20, 3.3, 53);

  const loader = new THREE.Group();
  equipment.add(loader);
  loader.position.set(52, surface(52, 37) + 0.3, 37);
  loader.rotation.y = -0.35;
  box(loader, [6, 1.6, 4], [0, 2.6, 0], yellow);
  box(loader, [2.7, 2.8, 3.1], [-1, 4.6, 0], glass);
  box(loader, [3, 0.3, 3.7], [-1, 6, 0], yellow);
  for (const x of [-2, 2]) for (const z of [-2.1, 2.1]) wheel(loader, x, 1.5, z);
  for (const z of [-1.3, 1.3]) {
    const arm = box(loader, [5, 0.5, 0.45], [4, 2.5, z], yellow);
    arm.rotation.z = -0.25;
  }
  box(loader, [2.7, 1.5, 5], [6, 1.1, 0], dark);
  box(loader, [3, 0.2, 5.3], [6.2, 0.4, 0], yellow);

  const sensors = new THREE.Group();
  world.add(sensors);
  const pickables = [];
  [[-65, -58], [66, -26], [-50, 10], [25, 40]].forEach(([x, z], index) => {
    const y = surface(x, z);
    const group = new THREE.Group();
    group.position.set(x, y, z);
    sensors.add(group);
    box(group, [0.5, 5, 0.5], [0, 2.5, 0], material(0xb9cecc));
    box(group, [3, 1.8, 1.5], [0, 5, 0], material(0x59bdb4));
    const orb = new THREE.Mesh(new THREE.SphereGeometry(1.6, 16, 12), new THREE.MeshBasicMaterial({color: 0x76eee1}));
    orb.position.y = 7;
    orb.userData = {name: "Station S0" + (index + 1), elevation: y.toFixed(1) + " m", temperature: (26.2 + index * 0.7).toFixed(1) + " °C"};
    group.add(orb);
    pickables.push(orb);
    const ring = new THREE.Mesh(new THREE.RingGeometry(3, 3.25, 48), new THREE.MeshBasicMaterial({color: 0x76eee1, side: THREE.DoubleSide, transparent: true, opacity: 0.6}));
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.25;
    group.add(ring);
  });

  const views = {
    overview: {position: [198, 155, 210], target: [0, 12, 0]},
    plan: {position: [0, 315, 0.5], target: [0, 12, 0]},
    face: {position: [25, 65, 280], target: [0, 21, -15]},
    detail: {position: [105, 62, 121], target: [30, 10, 42]}
  };
  function selectView(name) {
    const view = views[name];
    camera.position.set(...view.position);
    controls.target.set(...view.target);
    controls.update();
    document.querySelectorAll("[data-view]").forEach(button => {
      const active = button.dataset.view === name;
      button.classList.toggle("selected", active);
      button.setAttribute("aria-pressed", active);
    });
  }
  selectView("overview");
  document.querySelectorAll("[data-view]").forEach(button => button.addEventListener("click", () => selectView(button.dataset.view)));
  document.getElementById("geology").addEventListener("change", event => {
    setGeology(event.target.checked);
    document.querySelector(".legend").hidden = !event.target.checked;
  });
  document.getElementById("mesh").addEventListener("change", event => { wire.visible = event.target.checked; });
  document.getElementById("sensors").addEventListener("change", event => {
    sensors.visible = event.target.checked;
    document.getElementById("selection").hidden = true;
  });
  document.getElementById("equipment").addEventListener("change", event => { equipment.visible = event.target.checked; });
  document.getElementById("height").addEventListener("input", event => {
    world.scale.y = Number(event.target.value);
    document.getElementById("height-value").value = Number(event.target.value).toFixed(1) + "×";
  });
  document.getElementById("sun").addEventListener("input", event => {
    const progress = Number(event.target.value) / 100;
    sun.position.set(-150 + progress * 300, 55 + Math.sin(progress * Math.PI) * 150, 100);
    document.getElementById("sun-value").value = progress < 0.35 ? "Morning" : progress > 0.7 ? "Evening" : "Afternoon";
  });
  let moving = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const motion = document.getElementById("motion");
  function motionLabel() { motion.textContent = moving ? "Pause equipment motion" : "Resume equipment motion"; motion.setAttribute("aria-pressed", moving); }
  motionLabel();
  motion.addEventListener("click", () => { moving = !moving; motionLabel(); });
  document.getElementById("capture").addEventListener("click", () => {
    renderer.render(scene, camera);
    const link = document.createElement("a");
    // Include provenance in exported images as well as on the page.
    const canvas = document.createElement("canvas");
    canvas.width = renderer.domElement.width; canvas.height = renderer.domElement.height;
    const context = canvas.getContext("2d");
    context.fillStyle = "#15242e"; context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(renderer.domElement, 0, 0);
    context.fillStyle = "#ffd28c"; context.font = Math.max(15, canvas.width / 65) + "px sans-serif";
    context.fillText("BlasterOPT · Simulated bench · Fictional terrain and readings", 24, canvas.height - 24);
    link.download = "BlasterOPT-simulated-bench.png";
    link.href = canvas.toDataURL("image/png"); link.click();
  });
  const raycaster = new THREE.Raycaster();
  let pointerStart = {x: 0, y: 0};
  renderer.domElement.addEventListener("pointerdown", event => { pointerStart = {x: event.clientX, y: event.clientY}; });
  renderer.domElement.addEventListener("pointerup", event => {
    if (!sensors.visible || Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 5) return;
    const bounds = renderer.domElement.getBoundingClientRect();
    raycaster.setFromCamera(new THREE.Vector2((event.clientX - bounds.left) / bounds.width * 2 - 1, -(event.clientY - bounds.top) / bounds.height * 2 + 1), camera);
    const hit = raycaster.intersectObjects(pickables)[0];
    if (!hit) return;
    const data = hit.object.userData;
    document.getElementById("station-name").textContent = data.name;
    document.getElementById("station-elevation").textContent = data.elevation;
    document.getElementById("station-temperature").textContent = data.temperature;
    document.getElementById("selection").hidden = false;
  });
  document.getElementById("close-selection").addEventListener("click", () => { document.getElementById("selection").hidden = true; });
  new ResizeObserver(() => {
    const width = viewport.clientWidth, height = viewport.clientHeight;
    renderer.setSize(width, height, false);
    camera.aspect = width / height; camera.updateProjectionMatrix();
  }).observe(viewport);
  renderer.domElement.addEventListener("webglcontextlost", event => {
    event.preventDefault(); loading.style.display = "grid";
    loading.textContent = "The graphics context was interrupted. Reload to restore the scene.";
  });
  loading.style.display = "none";
  let lastTime = performance.now(), motionTime = 0;
  function animate(time) {
    requestAnimationFrame(animate);
    const delta = Math.min((time - lastTime) / 1000, 0.05);
    lastTime = time;
    if (moving && !document.hidden) motionTime += delta;
    truck.position.x = Math.sin(motionTime * 0.12) * 65;
    truck.rotation.y = Math.cos(motionTime * 0.12) < 0 ? Math.PI : 0;
    controls.update();
    const direction = camera.position.clone().sub(controls.target);
    document.querySelector(".compass span").style.transform = "rotate(" + (-Math.atan2(direction.x, direction.z) * 180 / Math.PI) + "deg)";
    renderer.render(scene, camera);
  }
  requestAnimationFrame(animate);
})();
