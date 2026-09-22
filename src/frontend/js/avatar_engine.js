/**
 * ISL Articulated Sign Avatar Engine
 * Renders smooth kinematic 2D/3D human and cyber sign avatars on HTML5 Canvas
 * Features:
 * - Smooth frame interpolation (linear & spring easing)
 * - Articulated fingers (5 per hand) & face expressions (eyes, brows, mouth)
 * - Multiple rendering styles: Humanoid, Cyber Mesh, Kinematic Skeleton
 * - Sequence playback controller with pause, scrub, speed modulation
 */

class SignAvatarRenderer {
  constructor(canvasId, options = {}) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) {
      console.warn(`[AvatarEngine] Canvas '${canvasId}' not found.`);
      return;
    }
    this.ctx = this.canvas.getContext('2d');
    this.style = options.style || 'humanoid'; // 'humanoid', 'cyber', 'skeleton'
    
    // Animation state
    this.isPlaying = false;
    this.speed = options.speed || 1.0;
    this.currentSignIndex = 0;
    this.currentFrameIndex = 0;
    this.frameProgress = 0.0; // 0.0 -> 1.0 interpolation between current & next frame
    this.sequence = []; // Array of sign objects: [{ word, frames: [...] }, ...]
    this.fps = 24;
    this.lastTimestamp = 0;
    this.animationFrameId = null;

    // Default Resting Pose
    this.defaultPose = {
      head: [0.5, 0.16],
      neck: [0.5, 0.25],
      torso: [0.5, 0.58],
      left_shoulder: [0.38, 0.28],
      right_shoulder: [0.62, 0.28],
      left_elbow: [0.30, 0.44],
      right_elbow: [0.70, 0.44],
      left_wrist: [0.32, 0.62],
      right_wrist: [0.68, 0.62],
      left_hand: [0.32, 0.68],
      right_hand: [0.68, 0.68],
      left_fingers: [1, 1, 1, 1, 1],
      right_fingers: [1, 1, 1, 1, 1],
      face: { eyebrows: 'neutral', eyes: 'open', mouth: 'neutral' }
    };

    this.currentPose = JSON.parse(JSON.stringify(this.defaultPose));
    this.targetPose = JSON.parse(JSON.stringify(this.defaultPose));

    this.onFrameUpdate = options.onFrameUpdate || null;
    this.onWordChange = options.onWordChange || null;
    this.onSequenceComplete = options.onSequenceComplete || null;

    this.resize();
    window.addEventListener('resize', () => this.resize());
    this.startRenderLoop();
  }

  resize() {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      this.canvas.width = rect.width * (window.devicePixelRatio || 1);
      this.canvas.height = rect.height * (window.devicePixelRatio || 1);
    }
  }

  setStyle(style) {
    this.style = style;
  }

  setSpeed(speed) {
    this.speed = Math.max(0.2, Math.min(3.0, speed));
  }

  loadSequence(sequenceData) {
    if (!Array.isArray(sequenceData) || sequenceData.length === 0) return;
    this.sequence = sequenceData;
    this.currentSignIndex = 0;
    this.currentFrameIndex = 0;
    this.frameProgress = 0.0;
    this.isPlaying = true;

    if (this.onWordChange) {
      this.onWordChange(this.sequence[0].word, 0);
    }
  }

  playSingleSign(signData) {
    this.loadSequence([signData]);
  }

  pause() {
    this.isPlaying = false;
  }

  resume() {
    this.isPlaying = true;
  }

  restart() {
    this.currentSignIndex = 0;
    this.currentFrameIndex = 0;
    this.frameProgress = 0.0;
    this.isPlaying = true;
  }

  startRenderLoop() {
    const loop = (timestamp) => {
      if (!this.lastTimestamp) this.lastTimestamp = timestamp;
      const deltaTime = (timestamp - this.lastTimestamp) / 1000;
      this.lastTimestamp = timestamp;

      this.update(deltaTime);
      this.render();

      this.animationFrameId = requestAnimationFrame(loop);
    };
    this.animationFrameId = requestAnimationFrame(loop);
  }

  update(deltaTime) {
    if (!this.isPlaying || this.sequence.length === 0) {
      // Gentle idle breathing animation
      const breath = Math.sin(Date.now() / 1200) * 0.008;
      this.currentPose.left_shoulder[1] = this.defaultPose.left_shoulder[1] + breath;
      this.currentPose.right_shoulder[1] = this.defaultPose.right_shoulder[1] + breath;
      return;
    }

    const currentSign = this.sequence[this.currentSignIndex];
    if (!currentSign || !currentSign.frames || currentSign.frames.length === 0) return;

    const frames = currentSign.frames;
    const currentFrame = frames[this.currentFrameIndex];
    const nextFrame = frames[(this.currentFrameIndex + 1) % frames.length];

    // Progress interpolation
    const frameRate = 12 * this.speed;
    this.frameProgress += deltaTime * frameRate;

    if (this.frameProgress >= 1.0) {
      this.frameProgress = 0.0;
      this.currentFrameIndex++;

      if (this.currentFrameIndex >= frames.length) {
        this.currentFrameIndex = 0;
        this.currentSignIndex++;

        if (this.currentSignIndex >= this.sequence.length) {
          // Completed full sentence
          this.currentSignIndex = 0;
          this.isPlaying = false;
          if (this.onSequenceComplete) this.onSequenceComplete();
          return;
        }

        if (this.onWordChange) {
          this.onWordChange(this.sequence[this.currentSignIndex].word, this.currentSignIndex);
        }
      }
    }

    // Smooth Cosine Interpolation
    const t = 0.5 * (1 - Math.cos(this.frameProgress * Math.PI));

    this.interpolatePose(currentFrame, nextFrame, t);

    if (this.onFrameUpdate) {
      const totalFrames = this.sequence.reduce((acc, s) => acc + (s.frames ? s.frames.length : 0), 0);
      let passedFrames = 0;
      for (let i = 0; i < this.currentSignIndex; i++) {
        passedFrames += this.sequence[i].frames ? this.sequence[i].frames.length : 0;
      }
      passedFrames += this.currentFrameIndex + this.frameProgress;
      this.onFrameUpdate(passedFrames / Math.max(1, totalFrames));
    }
  }

  interpolatePose(frameA, frameB, t) {
    const joints = [
      "head", "neck", "torso", "left_shoulder", "right_shoulder",
      "left_elbow", "right_elbow", "left_wrist", "right_wrist",
      "left_hand", "right_hand"
    ];

    joints.forEach(j => {
      const posA = frameA[j] || this.defaultPose[j];
      const posB = frameB[j] || this.defaultPose[j];
      this.currentPose[j] = [
        posA[0] + (posB[0] - posA[0]) * t,
        posA[1] + (posB[1] - posA[1]) * t
      ];
    });

    // Interpolate finger states
    ['left_fingers', 'right_fingers'].forEach(fKey => {
      const fA = frameA[fKey] || this.defaultPose[fKey];
      const fB = frameB[fKey] || this.defaultPose[fKey];
      this.currentPose[fKey] = fA.map((val, idx) => val + ((fB[idx] || val) - val) * t);
    });

    // Face attributes
    this.currentPose.face = frameB.face || frameA.face || this.defaultPose.face;
  }

  render() {
    if (!this.canvas || !this.ctx) return;
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Coordinate conversion helper
    const toScreen = ([nx, ny]) => [nx * w, ny * h];

    if (this.style === 'skeleton') {
      this.renderSkeleton(ctx, toScreen);
    } else if (this.style === 'cyber') {
      this.renderCyber(ctx, toScreen);
    } else {
      this.renderHumanoid(ctx, toScreen);
    }
  }

  /* --------------------------------------------------------------------------
   * 1. HUMANOID STYLE RENDERER (Clean Stylized Humanoid Avatar)
   * -------------------------------------------------------------------------- */
  renderHumanoid(ctx, toScreen) {
    const p = this.currentPose;
    const head = toScreen(p.head);
    const neck = toScreen(p.neck);
    const torso = toScreen(p.torso);
    const lSh = toScreen(p.left_shoulder);
    const rSh = toScreen(p.right_shoulder);
    const lEl = toScreen(p.left_elbow);
    const rEl = toScreen(p.right_elbow);
    const lWr = toScreen(p.left_wrist);
    const rWr = toScreen(p.right_wrist);
    const lHd = toScreen(p.left_hand);
    const rHd = toScreen(p.right_hand);

    const skinColor = '#e2b397';
    const skinShadow = '#c9967b';
    const shirtColor = '#4f46e5';
    const shirtAccent = '#3730a3';

    // 1. Torso / Shirt
    ctx.beginPath();
    ctx.moveTo(lSh[0], lSh[1]);
    ctx.lineTo(rSh[0], rSh[1]);
    ctx.lineTo(rSh[0] + 10, torso[1]);
    ctx.lineTo(lSh[0] - 10, torso[1]);
    ctx.closePath();
    const shirtGrad = ctx.createLinearGradient(lSh[0], lSh[1], torso[0], torso[1]);
    shirtGrad.addColorStop(0, shirtColor);
    shirtGrad.addColorStop(1, shirtAccent);
    ctx.fillStyle = shirtGrad;
    ctx.fill();

    // 2. Limbs (Arms)
    const drawLimb = (p1, p2, width, color) => {
      ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineCap = 'round';
      ctx.stroke();
    };

    // Upper arms (Sleeves)
    drawLimb(lSh, lEl, 28, shirtColor);
    drawLimb(rSh, rEl, 28, shirtColor);

    // Forearms (Skin)
    drawLimb(lEl, lWr, 20, skinColor);
    drawLimb(rEl, rWr, 20, skinColor);

    // 3. Hands & Articulated Fingers
    const drawHand = (wrist, handPos, fingers, isRight) => {
      // Palm
      ctx.beginPath();
      ctx.arc(handPos[0], handPos[1], 15, 0, Math.PI * 2);
      ctx.fillStyle = skinColor;
      ctx.fill();

      // Draw 5 articulated finger rays
      const dirX = handPos[0] - wrist[0];
      const dirY = handPos[1] - wrist[1];
      const angle = Math.atan2(dirY, dirX) || (isRight ? 0 : Math.PI);

      const fingerOffsets = [-0.5, -0.25, 0, 0.25, 0.5];
      fingers.forEach((extension, idx) => {
        const fAngle = angle + fingerOffsets[idx] * 0.8;
        const len = 14 * extension + 4;
        const fx = handPos[0] + Math.cos(fAngle) * len;
        const fy = handPos[1] + Math.sin(fAngle) * len;

        ctx.beginPath();
        ctx.moveTo(handPos[0], handPos[1]);
        ctx.lineTo(fx, fy);
        ctx.strokeStyle = skinShadow;
        ctx.lineWidth = 4;
        ctx.lineCap = 'round';
        ctx.stroke();
      });
    };

    drawHand(lWr, lHd, p.left_fingers, false);
    drawHand(rWr, rHd, p.right_fingers, true);

    // 4. Neck
    drawLimb(neck, [neck[0], neck[1] - 20], 18, skinShadow);

    // 5. Head
    ctx.beginPath();
    ctx.ellipse(head[0], head[1], 42, 50, 0, 0, Math.PI * 2);
    ctx.fillStyle = skinColor;
    ctx.fill();

    // Hair
    ctx.beginPath();
    ctx.ellipse(head[0], head[1] - 22, 44, 28, 0, Math.PI, Math.PI * 2);
    ctx.fillStyle = '#1e293b';
    ctx.fill();

    // Eyes
    const eyeY = head[1] - 4;
    const drawEye = (x, y) => {
      if (p.face && p.face.eyes === 'blink') {
        ctx.beginPath();
        ctx.moveTo(x - 6, y);
        ctx.lineTo(x + 6, y);
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 2.5;
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(x, y, 4.5, 0, Math.PI * 2);
        ctx.fillStyle = '#1e293b';
        ctx.fill();
        // Pupil shine
        ctx.beginPath();
        ctx.arc(x - 1.5, y - 1.5, 1.5, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
      }
    };
    drawEye(head[0] - 14, eyeY);
    drawEye(head[0] + 14, eyeY);

    // Eyebrows
    const browY = eyeY - 10 + (p.face && p.face.eyebrows === 'raised' ? -4 : 0);
    ctx.beginPath();
    ctx.moveTo(head[0] - 20, browY);
    ctx.lineTo(head[0] - 8, browY - (p.face && p.face.eyebrows === 'furrowed' ? -3 : 2));
    ctx.moveTo(head[0] + 8, browY - (p.face && p.face.eyebrows === 'furrowed' ? -3 : 2));
    ctx.lineTo(head[0] + 20, browY);
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 3;
    ctx.stroke();

    // Mouth / Smile
    const mouthY = head[1] + 22;
    ctx.beginPath();
    if (p.face && p.face.mouth === 'smile') {
      ctx.arc(head[0], mouthY - 4, 12, 0.2, Math.PI - 0.2);
    } else if (p.face && p.face.mouth === 'open') {
      ctx.ellipse(head[0], mouthY, 8, 5, 0, 0, Math.PI * 2);
    } else {
      ctx.moveTo(head[0] - 8, mouthY);
      ctx.lineTo(head[0] + 8, mouthY);
    }
    ctx.strokeStyle = '#991b1b';
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }

  /* --------------------------------------------------------------------------
   * 2. CYBER MESH STYLE RENDERER (Neon HUD Avatar)
   * -------------------------------------------------------------------------- */
  renderCyber(ctx, toScreen) {
    const p = this.currentPose;
    const joints = [
      "head", "neck", "torso", "left_shoulder", "right_shoulder",
      "left_elbow", "right_elbow", "left_wrist", "right_wrist",
      "left_hand", "right_hand"
    ];

    const bones = [
      ["head", "neck"], ["neck", "torso"],
      ["neck", "left_shoulder"], ["left_shoulder", "left_elbow"],
      ["left_elbow", "left_wrist"], ["left_wrist", "left_hand"],
      ["neck", "right_shoulder"], ["right_shoulder", "right_elbow"],
      ["right_elbow", "right_wrist"], ["right_wrist", "right_hand"]
    ];

    // Neon Glow lines
    ctx.shadowBlur = 14;
    ctx.shadowColor = '#06b6d4';
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 4;

    bones.forEach(([j1, j2]) => {
      const p1 = toScreen(p[j1]);
      const p2 = toScreen(p[j2]);
      ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);
      ctx.stroke();
    });

    // Glowing Joint Orbs
    joints.forEach(j => {
      const pos = toScreen(p[j]);
      ctx.beginPath();
      ctx.arc(pos[0], pos[1], j === 'head' ? 24 : 7, 0, Math.PI * 2);
      ctx.fillStyle = '#a855f7';
      ctx.shadowColor = '#a855f7';
      ctx.fill();
    });

    ctx.shadowBlur = 0; // Reset shadow
  }

  /* --------------------------------------------------------------------------
   * 3. SKELETON BONES STYLE RENDERER
   * -------------------------------------------------------------------------- */
  renderSkeleton(ctx, toScreen) {
    const p = this.currentPose;
    const bones = [
      ["head", "neck"], ["neck", "torso"],
      ["neck", "left_shoulder"], ["left_shoulder", "left_elbow"],
      ["left_elbow", "left_wrist"], ["left_wrist", "left_hand"],
      ["neck", "right_shoulder"], ["right_shoulder", "right_elbow"],
      ["right_elbow", "right_wrist"], ["right_wrist", "right_hand"]
    ];

    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';

    bones.forEach(([j1, j2]) => {
      const p1 = toScreen(p[j1]);
      const p2 = toScreen(p[j2]);
      ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(p1[0], p1[1], 5, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
    });
  }
}
