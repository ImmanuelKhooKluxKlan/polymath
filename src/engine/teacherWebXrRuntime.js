import { clamp, sanitizeTeacherArSettings, teacherArImageUrl } from './teacherArEngine.js';

const VERTEX_SHADER = `
  attribute vec3 a_position;
  attribute vec2 a_uv;
  uniform mat4 u_model;
  uniform mat4 u_view;
  uniform mat4 u_projection;
  uniform vec4 u_uv_rect;
  varying vec2 v_uv;
  void main() {
    v_uv = mix(u_uv_rect.xy, u_uv_rect.zw, a_uv);
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
  }
`;

const FRAGMENT_SHADER = `
  precision mediump float;
  uniform sampler2D u_texture;
  uniform float u_opacity;
  varying vec2 v_uv;
  void main() {
    vec4 color = texture2D(u_texture, v_uv);
    if (color.a < 0.02) discard;
    gl_FragColor = vec4(color.rgb, color.a * u_opacity);
  }
`;

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const detail = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`AR renderer shader failed: ${detail}`);
  }
  return shader;
}

function createProgram(gl) {
  const program = gl.createProgram();
  const vertex = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const detail = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`AR renderer link failed: ${detail}`);
  }
  return program;
}

function modelMatrix(position, viewerPosition, scale) {
  const height = 1.66 * scale;
  const width = height * 0.75;
  const yaw = Math.atan2(
    viewerPosition.x - position.x,
    viewerPosition.z - position.z,
  );
  const cosine = Math.cos(yaw);
  const sine = Math.sin(yaw);
  return new Float32Array([
    cosine * width, 0, -sine * width, 0,
    0, height, 0, 0,
    sine, 0, cosine, 0,
    position.x, position.y, position.z, 1,
  ]);
}

function aheadOfViewer(viewerTransform, distance = 1.8) {
  const matrix = viewerTransform.matrix;
  const position = viewerTransform.position;
  return {
    x: position.x - matrix[8] * distance,
    y: Math.max(0, position.y - 1.55),
    z: position.z - matrix[10] * distance,
  };
}

function imagePromise(url) {
  return new Promise((resolve, reject) => {
    const image = new globalThis.Image();
    image.decoding = 'async';
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('The AR teacher artwork could not be loaded.'));
    image.src = url;
  });
}

export class TeacherWebXrRuntime {
  constructor({ canvas, overlayRoot, settings, onStatus, onEnded, onError } = {}) {
    this.canvas = canvas;
    this.overlayRoot = overlayRoot;
    this.settings = sanitizeTeacherArSettings(settings);
    this.onStatus = onStatus || (() => {});
    this.onEnded = onEnded || (() => {});
    this.onError = onError || (() => {});
    this.session = null;
    this.gl = null;
    this.program = null;
    this.texture = null;
    this.referenceSpace = null;
    this.viewerSpace = null;
    this.hitTestSource = null;
    this.latestHitResult = null;
    this.previewPosition = null;
    this.placement = null;
    this.anchor = null;
    this.ended = false;
    this.boundFrame = this.onXrFrame.bind(this);
    this.boundSelect = this.placeCurrent.bind(this);
    this.boundEnd = this.handleSessionEnd.bind(this);
  }

  async start() {
    if (!navigator.xr?.requestSession) throw new Error('This browser does not expose WebXR.');
    if (!this.canvas) throw new Error('The AR rendering canvas is missing.');

    this.onStatus({ phase: 'requesting', message: 'Waiting for the glasses…' });
    const options = {
      optionalFeatures: ['local-floor', 'hit-test', 'anchors', 'dom-overlay'],
    };
    if (this.overlayRoot) options.domOverlay = { root: this.overlayRoot };
    this.session = await navigator.xr.requestSession('immersive-ar', options);
    this.session.addEventListener('end', this.boundEnd);
    this.session.addEventListener('select', this.boundSelect);
    try {
      const gl = this.canvas.getContext('webgl', {
        alpha: true,
        antialias: true,
        depth: true,
        premultipliedAlpha: false,
        xrCompatible: true,
      });
      if (!gl) throw new Error('WebGL is unavailable on this AR browser.');
      this.gl = gl;
      await gl.makeXRCompatible?.();
      if (!globalThis.XRWebGLLayer) throw new Error('This browser has WebXR but no XR WebGL layer.');
      this.session.updateRenderState({
        baseLayer: new globalThis.XRWebGLLayer(this.session, gl, { alpha: true, antialias: true }),
        depthNear: 0.05,
        depthFar: 30,
      });

      this.referenceSpace = await this.session.requestReferenceSpace('local');
      this.viewerSpace = await this.session.requestReferenceSpace('viewer');
      if (this.session.requestHitTestSource) {
        try {
          this.hitTestSource = await this.session.requestHitTestSource({ space: this.viewerSpace });
        } catch {
          this.hitTestSource = null;
        }
      }

      await this.initializeRenderer();
      this.onStatus({
        phase: 'scanning',
        message: this.hitTestSource
          ? 'Look at the floor beside the piano, then tap to place her.'
          : 'Surface scanning is unavailable. Tap to place her in front of you.',
        hitTest: Boolean(this.hitTestSource),
      });
      this.session.requestAnimationFrame(this.boundFrame);
      return this;
    } catch (error) {
      await this.session?.end?.().catch(() => {});
      throw error;
    }
  }

  async initializeRenderer() {
    const gl = this.gl;
    this.program = createProgram(gl);
    gl.useProgram(this.program);
    const vertices = new Float32Array([
      -0.5, 0, 0, 0, 0,
       0.5, 0, 0, 1, 0,
      -0.5, 1, 0, 0, 1,
      -0.5, 1, 0, 0, 1,
       0.5, 0, 0, 1, 0,
       0.5, 1, 0, 1, 1,
    ]);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    const stride = 5 * Float32Array.BYTES_PER_ELEMENT;
    const positionLocation = gl.getAttribLocation(this.program, 'a_position');
    const uvLocation = gl.getAttribLocation(this.program, 'a_uv');
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(uvLocation);
    gl.vertexAttribPointer(uvLocation, 2, gl.FLOAT, false, stride, 3 * Float32Array.BYTES_PER_ELEMENT);
    this.uniforms = {
      model: gl.getUniformLocation(this.program, 'u_model'),
      view: gl.getUniformLocation(this.program, 'u_view'),
      projection: gl.getUniformLocation(this.program, 'u_projection'),
      uvRect: gl.getUniformLocation(this.program, 'u_uv_rect'),
      opacity: gl.getUniformLocation(this.program, 'u_opacity'),
    };
    gl.uniform1i(gl.getUniformLocation(this.program, 'u_texture'), 0);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.CULL_FACE);
    gl.enable(gl.DEPTH_TEST);
    await this.loadTexture(this.settings.style);
  }

  async loadTexture(style) {
    const gl = this.gl;
    if (!gl) return;
    const image = await imagePromise(teacherArImageUrl(style));
    if (!this.gl || this.ended) return;
    if (this.texture) gl.deleteTexture(this.texture);
    this.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
  }

  setSettings(next) {
    const previousStyle = this.settings.style;
    this.settings = sanitizeTeacherArSettings({ ...this.settings, ...next });
    if (previousStyle !== this.settings.style) {
      this.loadTexture(this.settings.style).catch(this.onError);
    }
  }

  async placeCurrent() {
    if (!this.previewPosition) return;
    this.anchor?.delete?.();
    this.anchor = null;
    this.placement = { ...this.previewPosition };
    if (this.latestHitResult?.createAnchor) {
      try {
        this.anchor = await this.latestHitResult.createAnchor();
      } catch {
        this.anchor = null;
      }
    }
    this.onStatus({ phase: 'placed', message: 'Teacher placed. Use the controls to fine-tune her position.' });
  }

  move({ x = 0, y = 0, z = 0 } = {}) {
    if (!this.placement) this.placement = { ...(this.previewPosition || { x: 0, y: 0, z: -1.8 }) };
    this.anchor?.delete?.();
    this.anchor = null;
    this.placement = {
      x: this.placement.x + clamp(Number(x) || 0, -0.5, 0.5),
      y: Math.max(0, this.placement.y + clamp(Number(y) || 0, -0.5, 0.5)),
      z: this.placement.z + clamp(Number(z) || 0, -0.5, 0.5),
    };
  }

  resetPlacement() {
    this.anchor?.delete?.();
    this.anchor = null;
    this.placement = null;
    this.onStatus({ phase: 'scanning', message: 'Look at the floor beside the piano, then tap to place her again.' });
  }

  async end() {
    if (this.session) await this.session.end().catch(() => {});
  }

  handleSessionEnd() {
    if (this.ended) return;
    this.ended = true;
    this.hitTestSource?.cancel?.();
    this.anchor?.delete?.();
    this.session?.removeEventListener('select', this.boundSelect);
    this.session?.removeEventListener('end', this.boundEnd);
    this.onEnded();
  }

  onXrFrame(_time, frame) {
    if (!this.session || this.ended) return;
    this.session.requestAnimationFrame(this.boundFrame);
    const pose = frame.getViewerPose(this.referenceSpace);
    if (!pose) return;

    if (this.anchor) {
      const anchorPose = frame.getPose(this.anchor.anchorSpace, this.referenceSpace);
      if (anchorPose) this.placement = { ...anchorPose.transform.position };
    }

    if (!this.placement) {
      this.latestHitResult = null;
      if (this.hitTestSource) {
        const results = frame.getHitTestResults(this.hitTestSource);
        this.latestHitResult = results[0] || null;
        const hitPose = this.latestHitResult?.getPose(this.referenceSpace);
        if (hitPose) this.previewPosition = { ...hitPose.transform.position };
      }
      if (!this.previewPosition) this.previewPosition = aheadOfViewer(pose.transform);
    }

    const position = this.placement || this.previewPosition;
    if (!position || !this.texture) return;
    const gl = this.gl;
    const layer = this.session.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.uniformMatrix4fv(this.uniforms.model, false, modelMatrix(position, pose.transform.position, this.settings.scale));
    gl.uniform4f(this.uniforms.uvRect, this.settings.speaking ? 0.5 : 0, 0, this.settings.speaking ? 1 : 0.5, 1);
    gl.uniform1f(this.uniforms.opacity, this.placement ? this.settings.opacity : this.settings.opacity * 0.58);

    for (const view of pose.views) {
      const viewport = layer.getViewport(view);
      gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
      gl.uniformMatrix4fv(this.uniforms.view, false, view.transform.inverse.matrix);
      gl.uniformMatrix4fv(this.uniforms.projection, false, view.projectionMatrix);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
  }
}
