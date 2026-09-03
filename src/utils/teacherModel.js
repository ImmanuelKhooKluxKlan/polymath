const GLB_LIMIT_BYTES = 25 * 1024 * 1024;

export async function validateTeacherGlbFile(file) {
  if (!(file instanceof Blob)) throw new Error('Choose a rigged GLB model.');
  if (!String(file.name || '').toLowerCase().endsWith('.glb')) {
    throw new Error('The 3D model must be a .glb file.');
  }
  if (file.size > GLB_LIMIT_BYTES) throw new Error('The 3D model must be 25 MB or smaller.');
  if (file.size < 20) throw new Error('The selected GLB file is incomplete.');
  const header = await file.slice(0, 20).arrayBuffer();
  const bytes = new Uint8Array(header);
  const view = new DataView(header);
  const magic = String.fromCharCode(...bytes.slice(0, 4));
  if (magic !== 'glTF' || view.getUint32(4, true) !== 2 || view.getUint32(8, true) !== file.size) {
    throw new Error('Choose a valid binary glTF 2.0 (.glb) model.');
  }
  return file;
}

