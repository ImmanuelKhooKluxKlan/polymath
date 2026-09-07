# Polymath AR Teacher Blueprint

Status: working engineering foundation. It is not yet a finished human-like 3D teacher.

## 1. What was added

Polymath now has one AR studio with two progressively enhanced runtimes:

```text
Learn lesson
   |
   +-- AR glasses button
          |
          +-- Spatial WebXR runtime
          |      Glasses/headset cameras understand surfaces
          |      -> WebXR hit test returns a floor position
          |      -> user taps/selects
          |      -> optional XR anchor stabilises the position
          |      -> WebGL draws one teacher view per eye
          |
          +-- Camera simulator
                 Phone rear camera stays local
                 -> user taps the floor in the preview
                 -> DOM teacher is positioned at that point
                 -> same size/style controls can be tested cheaply
```

Routes:

- `#teacher-ar?style=regular` opens the AR studio.
- `#teacher-ar?style=regular&simulate=1` opens the camera simulator.

Important files:

- `src/components/TeacherArControls.jsx`: Learn-page launcher and live state bridge.
- `src/pages/TeacherArPage.jsx`: AR studio, setup, calibration controls and errors.
- `src/components/TeacherArSimulator.jsx`: phone/camera fallback.
- `src/engine/teacherArEngine.js`: capability detection and input sanitising.
- `src/engine/teacherWebXrRuntime.js`: WebXR session, hit testing, anchors and stereoscopic WebGL renderer.
- `scripts/performance/teacherAr.test.mjs`: safety and capability tests.

## 2. What the spatial runtime really does

### Capability check

The page asks the browser whether `immersive-ar` is supported. This is a read-only check. It does not silently start cameras or tracking.

### Permission and session start

The user must press **Start spatial AR**. WebXR requires a deliberate user action. The browser/glasses operating system owns the permission prompt.

The runtime requests these features progressively:

- `local-floor`: a coordinate system related to the physical floor when available.
- `hit-test`: rays intersect known real-world surfaces.
- `anchors`: a placed teacher can remain stable as the device improves its map.
- `dom-overlay`: small HTML controls remain usable in the immersive session.

They are optional so one missing feature does not unnecessarily reject the complete session. If hit testing is absent, Polymath places a preview about 1.8 metres in front of the viewer.

### Placement

While scanning, the teacher is semi-transparent. The user looks at the floor beside the piano and presses a controller, taps, or uses the headset's select gesture. Polymath records the returned 3D position. If anchors are supported, it asks the device to maintain an anchor there.

### Rendering

The runtime uses an alpha-enabled WebGL layer. Every XR frame contains one view for each eye. For each view it:

1. Reads the eye's view and projection matrices.
2. Rotates the teacher billboard toward the viewer.
3. Draws the teacher using the correct eye camera.
4. Leaves the background transparent so the real room remains visible.

This makes the existing 2D artwork look spatially positioned. It is not yet a rigged 3D human and therefore cannot walk around furniture or turn naturally in profile.

## 3. Piano calibration

The current calibration is deliberately manual and reliable:

1. Stand or sit in the normal playing position.
2. Look at the floor to the left or right of the piano.
3. Select **Place here** or use the headset select gesture.
4. Adjust Left, Right, Away, Closer and Size.
5. Re-scan whenever the piano or chair moves.

The `Piano side` setting is already reserved in the state contract for future coaching layouts. Automatic piano recognition must not be claimed until a vision model has been deployed and validated. WebXR generally provides spatial poses and surfaces; it does not automatically tell our JavaScript that an object is a piano.

## 4. Privacy boundary

Spatial mode:

- The XR operating system uses its cameras to calculate poses and hit-test results.
- Polymath receives coordinate matrices and surface intersections.
- This code does not request or upload raw XR camera frames.

Camera simulator:

- Camera permission is requested only after **Allow camera** is pressed.
- Audio is explicitly disabled.
- The stream is attached only to the local video element.
- Camera tracks stop when the camera is switched off or the page closes.
- No frame is uploaded by the simulator.

Any future object-recognition feature must have a second explicit consent screen and a visible indicator explaining whether frames stay local or are sent to a server.

## 5. Alibaba purchase checklist

Do not buy based only on the words “AR glasses.” Many inexpensive products are external displays. A display-only product can show a floating flat screen but cannot place a teacher beside a real piano.

Ask the seller to confirm all of the following in writing:

1. **WebXR immersive-ar** works in the included browser.
2. **6DoF** tracking is supported. 3DoF only knows head rotation, not position in the room.
3. The user sees the physical room through transparent optics or camera passthrough.
4. The browser supports WebGL and real-world hit testing or plane detection.
5. There is a select method: hand tracking, controller, touchpad or gaze click.
6. The device has a standalone browser, or clearly state which phone/compute unit runs it.
7. The seller can demonstrate an official WebXR immersive AR sample, not only screen mirroring.

Useful seller question:

> Please record this exact unit opening an HTTPS WebXR immersive-ar hit-test demo, placing an object on a real floor, then moving around it while it stays anchored. Please also show the browser name and firmware version.

Reject vague answers such as “supports AR apps,” “Android compatible,” “has a camera,” or “supports 3D movies.” None proves WebXR, 6DoF or anchors.

## 6. Testing without glasses

1. Run the frontend and backend normally.
2. Open Learn and unlock the virtual teacher.
3. Press **AR glasses**.
4. Select **Test with camera**.
5. Allow the rear camera.
6. Tap the floor beside a piano to move the teacher.
7. Test portrait and landscape, size changes and camera shutdown.

The camera simulator tests interface, placement and visibility. It cannot prove stereoscopic rendering, tracking latency, hit testing or anchor quality. Those require actual hardware.

## 7. Hardware acceptance test

Run this before accepting or ordering many units:

- Browser reports spatial WebXR ready.
- Session starts from one press without sideloaded browser hacks.
- Left and right eyes both render correctly.
- Floor hit testing finds a stable location within ten seconds.
- Teacher remains within about a hand width of the selected location while the user walks around.
- Select input works while seated at a piano.
- Controls do not accidentally move the teacher.
- Frame rate remains comfortable for a five-minute lesson.
- Device temperature and battery remain acceptable for a 30-minute lesson.
- Camera/session ends immediately when AR is closed.

## 8. Next engineering stages

### Stage A — current foundation

- WebXR/camera capability detection.
- Stereoscopic teacher billboard.
- Hit-test placement with optional anchors.
- Manual piano-side calibration.
- Camera fallback.
- Live teacher style/speaking state through `BroadcastChannel`.
- No raw camera upload.

### Stage B — 3D teacher

- Replace the billboard with a licensed, rigged glTF/GLB adult teacher.
- Add body animation, lip synchronisation and pointer gestures.
- Use level-of-detail models for weaker hardware.
- Add animation-state tests so speech and gestures do not conflict.

### Stage C — piano understanding

- Calibrate keyboard corners using explicit user taps first.
- Map the 88 physical keys to a keyboard coordinate system.
- Fuse microphone pitch/onset estimates with hand/key observations.
- Measure accuracy separately for pitch, timing, duration and hand position.
- Keep a confidence threshold; the teacher should ask to re-calibrate instead of inventing a correction.

### Stage D — spatial coaching

- Draw a highlight above the expected physical key.
- Show hold-duration rings and upcoming-note guides.
- Make the teacher point only when the physical key mapping is confident.
- Add occlusion where supported so the teacher can appear behind a piano rather than always on top.

### Stage E — production safety and analytics

- Explicit camera/vision consent and persistent recording indicator.
- Local processing where practical.
- No background capture.
- Per-device performance telemetry with no camera images.
- Crash, thermal, frame-time and anchor-drift dashboards.
- Accessibility mode, seated boundary and reduced motion.

## 9. Known limitations

- A 2D teacher cannot convincingly perform complex body turns.
- Hardware/browser WebXR modules vary. Optional-feature fallbacks are intentional.
- The current build does not identify the piano automatically.
- The current build does not obtain raw WebXR camera frames.
- The camera simulator is screen-space AR, not genuine world tracking.
- Real glasses must be tested before making a purchasing decision.
- `isSessionSupported('immersive-ar')` is advisory; the final session can still fail if hardware is busy or permission is denied.

## 10. Why this architecture is maintainable

The UI does not contain hardware-specific tracking code. Capability checks, pure input rules, WebXR rendering and camera simulation are separate modules. We can later replace only the renderer with a 3D engine without rewriting the Learn lesson, conversation engine or coaching logic.
