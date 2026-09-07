# Polymath teacher projection

## What exists now

### Wall mode

The lesson device remains the controller. A second browser surface renders only
the teacher, speaking pose and latest measured coaching caption on a black
background. The output can be placed on:

- an HDMI or USB-C projector;
- an extended desktop monitor or television;
- a compatible wireless display through the browser Presentation API; or
- another projector-connected computer using the expiring display link.

The wall still requires physical projection hardware. A phone or website cannot
emit a wall-sized image without a projector, television or spatial display.

### Pyramid mode

Pyramid mode renders four rotated copies around a black center. A transparent
four-sided reflector placed over the screen produces a Pepper's Ghost floating
image illusion. It is inexpensive and useful for prototyping, but it is not a
volumetric hologram and it has a narrow useful viewing area.

## Communication flow

```text
Learn page / teacher controller
  ├─ same browser: BroadcastChannel (near-instant)
  ├─ compatible display: W3C Presentation API
  └─ separate device: HTTPS state sync
                         │
                         ▼
             expiring projection session
                         │
                         ▼
              teacher-only display page
```

Only these bounded values cross the projection channel:

- `mode`: `wall` or `hologram`;
- `style`: `regular` or `bikini`;
- `speaking`: boolean;
- `visible`: boolean; and
- `caption`: at most 280 characters.

No camera frame, microphone sample, password, account token or chat transcript
is included. The display secret is generated from 32 random bytes, kept in the
URL fragment, stored server-side only as a SHA-256 hash and expires after two
hours. Production sessions use their own PostgreSQL table so both AWS regions
can resolve the same display link. Local development uses process memory.

## Operating it

1. Open Learn and unlock the virtual teacher.
2. Press **Project teacher**.
3. Choose **Wall** or **Pyramid**.
4. For an attached projector, press **Open display**, move the new window to the
   projector and press **Enter fullscreen**.
5. For another device, copy the display link and open it in a browser connected
   to the projector.
6. Where supported, **Wireless display** asks the browser to choose a receiver.
7. Press **End** to invalidate the session immediately.

## What a true hologram requires later

The existing teacher is a 2D two-pose sprite. A real light-field display needs a
rigged 3D teacher, facial blendshapes, body animation and many rendered camera
views per frame. Looking Glass, for example, combines multiple views into a
quilt and applies display-specific calibration. Its WebXR path also requires
compatible display hardware and Looking Glass Bridge.

Recommended later phases:

1. Build a licensed rigged 3D adult teacher in Blender.
2. Export glTF/GLB with idle, speaking, pointing and piano-hand animations.
3. Render it with Three.js for normal wall and WebXR/AR modes.
4. Add Looking Glass WebXR only after obtaining test hardware.
5. Add face/lip animation from the teacher speech phonemes.
6. Move cross-device state from one-second polling to a managed WebSocket
   service when concurrent projection usage justifies it.

References:

- W3C Presentation API: https://www.w3.org/TR/presentation-api/
- Looking Glass WebXR: https://github.com/Looking-Glass/looking-glass-webxr
