const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');

const testDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'polymath-virtual-teachers-test-'));
process.env.POLYMATH_DATA_DIR = testDataDir;
process.env.NODE_ENV = 'test';
process.env.REGISTRATION_OTP_TEST_CODE = '123456';
process.env.MUSCRIPTOR_ENABLED = 'false';
process.env.ADMIN_EMAILS = 'character-admin@example.test';
process.env.DATABASE_URL = '';
process.env.ARTIFACT_S3_BUCKET = '';

const { app } = require('./server');

function testRiggedGlb() {
  const document = {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [
      { name: 'Hips', mesh: 0, skin: 0, children: [1, 3, 4, 5, 6] },
      { name: 'Spine', children: [2] },
      { name: 'Head' },
      { name: 'LeftArm' },
      { name: 'RightArm' },
      { name: 'LeftUpLeg' },
      { name: 'RightUpLeg' },
    ],
    skins: [{ joints: [0, 1, 2, 3, 4, 5, 6] }],
    meshes: [{ primitives: [] }],
  };
  const source = Buffer.from(JSON.stringify(document));
  const paddedLength = Math.ceil(source.length / 4) * 4;
  const json = Buffer.alloc(paddedLength, 0x20);
  source.copy(json);
  const glb = Buffer.alloc(20 + json.length);
  glb.write('glTF', 0, 'ascii');
  glb.writeUInt32LE(2, 4);
  glb.writeUInt32LE(glb.length, 8);
  glb.writeUInt32LE(json.length, 12);
  glb.writeUInt32LE(0x4e4f534a, 16);
  json.copy(glb, 20);
  return glb;
}

test('administrators can create, edit, price, age-gate, hide, restore, and delete virtual teachers', async (context) => {
  const server = await new Promise((resolve) => {
    const instance = app.listen(0, '127.0.0.1', () => resolve(instance));
  });
  context.after(async () => {
    await new Promise((resolve) => server.close(resolve));
    fs.rmSync(testDataDir, { recursive: true, force: true });
  });
  const baseUrl = `http://127.0.0.1:${server.address().port}`;

  async function jsonApi(pathname, { method = 'GET', token = '', body } = {}) {
    const response = await fetch(`${baseUrl}${pathname}`, {
      method,
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(body ? { 'Content-Type': 'application/json' } : {}),
      },
      ...(body ? { body: JSON.stringify(body) } : {}),
    });
    return { status: response.status, data: await response.json() };
  }

  const challenge = await jsonApi('/api/auth/register/otp', {
    method: 'POST',
    body: { channel: 'email', email: 'character-admin@example.test' },
  });
  assert.equal(challenge.status, 202);
  const registration = await jsonApi('/api/auth/register', {
    method: 'POST',
    body: {
      name: 'Character Admin',
      email: 'character-admin@example.test',
      password: 'CharacterTest123',
      challengeId: challenge.data.challengeId,
      verificationCode: '123456',
    },
  });
  assert.equal(registration.status, 201);
  assert.equal(registration.data.user.admin, true);
  const token = registration.data.token;

  const initialAdminCatalog = await jsonApi('/api/admin/virtual-teachers', { token });
  assert.equal(initialAdminCatalog.status, 200);
  assert.equal(initialAdminCatalog.data.catalogVersion, 2);
  assert.deepEqual(
    initialAdminCatalog.data.catalog.filter((character) => character.builtIn).map((character) => character.id).sort(),
    ['anakin', 'aria', 'mace', 'nova', 'taylor'],
  );
  assert.equal(initialAdminCatalog.data.catalog.find((character) => character.id === 'nova').adultCompanionEnabled, true);

  const onePixelPng = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
    'base64',
  );
  const form = new FormData();
  form.append('name', 'Lyra');
  form.append('title', 'Rhythm coach');
  form.append('description', 'Keeps practice sessions steady and clear.');
  form.append('voice', 'Calm');
  form.append('armTone', 'dark');
  form.append('minimumAge', '16');
  form.append('pricePer30MinutesMcoins', '7.5');
  form.append('active', 'true');
  form.append('image', new Blob([onePixelPng], { type: 'image/png' }), 'aria.png');
  const riggedGlb = testRiggedGlb();
  form.append('model', new Blob([riggedGlb], { type: 'model/gltf-binary' }), 'aria.glb');
  const createdResponse = await fetch(`${baseUrl}/api/admin/virtual-teachers`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  });
  const created = await createdResponse.json();
  assert.equal(createdResponse.status, 201);
  assert.equal(created.character.name, 'Lyra');
  assert.equal(created.character.armTone, 'dark');
  assert.equal(created.character.minimumAge, 16);
  assert.equal(created.character.requiresAdultConfirmation, false);
  assert.equal(created.character.adultCompanionEnabled, false);
  assert.equal(created.character.pricePer30MinutesMcoins, 7.5);
  assert.equal(created.character.effectivePricePer30MinutesMcoins, 7.5);
  assert.equal(created.character.rig.jointCount, 7);
  assert.match(created.character.modelPath, /\/model\?/);

  const publicList = await jsonApi('/api/virtual-teachers');
  assert.equal(publicList.status, 200);
  assert.equal(publicList.data.catalogVersion, 2);
  assert.deepEqual(publicList.data.characters.map((character) => character.name), ['Lyra']);
  assert.equal(publicList.data.catalog.length, 6);
  const imageResponse = await fetch(`${baseUrl}${created.character.imagePath}`);
  assert.equal(imageResponse.status, 200);
  assert.equal(imageResponse.headers.get('content-type'), 'image/png');
  assert.deepEqual(Buffer.from(await imageResponse.arrayBuffer()), onePixelPng);
  const modelResponse = await fetch(`${baseUrl}${created.character.modelPath}`);
  assert.equal(modelResponse.status, 200);
  assert.equal(modelResponse.headers.get('content-type'), 'model/gltf-binary');
  assert.deepEqual(Buffer.from(await modelResponse.arrayBuffer()), riggedGlb);

  const customUpdateForm = new FormData();
  customUpdateForm.append('name', 'Lyra Renamed');
  customUpdateForm.append('minimumAge', '21');
  customUpdateForm.append('pricePer30MinutesMcoins', '12.25');
  const customUpdateResponse = await fetch(`${baseUrl}/api/admin/virtual-teachers/${created.character.id}`, {
    method: 'PATCH',
    headers: { Authorization: `Bearer ${token}` },
    body: customUpdateForm,
  });
  const customUpdate = await customUpdateResponse.json();
  assert.equal(customUpdateResponse.status, 200);
  assert.equal(customUpdate.character.name, 'Lyra Renamed');
  assert.equal(customUpdate.character.minimumAge, 21);
  assert.equal(customUpdate.character.requiresAdultConfirmation, true);
  assert.equal(customUpdate.character.pricePer30MinutesMcoins, 12.25);

  const builtInUpdateForm = new FormData();
  builtInUpdateForm.append('name', 'Anakin Prime');
  builtInUpdateForm.append('minimumAge', '13');
  builtInUpdateForm.append('pricePer30MinutesMcoins', '2.25');
  const builtInUpdateResponse = await fetch(`${baseUrl}/api/admin/virtual-teachers/anakin`, {
    method: 'PATCH',
    headers: { Authorization: `Bearer ${token}` },
    body: builtInUpdateForm,
  });
  const builtInUpdate = await builtInUpdateResponse.json();
  assert.equal(builtInUpdateResponse.status, 200);
  assert.equal(builtInUpdate.character.name, 'Anakin Prime');
  assert.equal(builtInUpdate.character.minimumAge, 13);
  assert.equal(builtInUpdate.character.adultCompanionEnabled, false);
  assert.equal(builtInUpdate.character.pricePer30MinutesMcoins, 2.25);

  const builtInDelete = await jsonApi('/api/admin/virtual-teachers/anakin', { method: 'DELETE', token });
  assert.equal(builtInDelete.status, 200);
  assert.equal(builtInDelete.data.disabled, true);
  const hiddenPublicList = await jsonApi('/api/virtual-teachers');
  assert.equal(hiddenPublicList.data.catalog.some((character) => character.id === 'anakin'), false);
  const hiddenAdminList = await jsonApi('/api/admin/virtual-teachers', { token });
  assert.equal(hiddenAdminList.data.catalog.find((character) => character.id === 'anakin').active, false);

  const restoreForm = new FormData();
  restoreForm.append('active', 'true');
  const restoreResponse = await fetch(`${baseUrl}/api/admin/virtual-teachers/anakin`, {
    method: 'PATCH',
    headers: { Authorization: `Bearer ${token}` },
    body: restoreForm,
  });
  assert.equal(restoreResponse.status, 200);
  const restoredPublicList = await jsonApi('/api/virtual-teachers');
  assert.equal(restoredPublicList.data.catalog.find((character) => character.id === 'anakin').name, 'Anakin Prime');

  const deleted = await jsonApi(`/api/admin/virtual-teachers/${created.character.id}`, { method: 'DELETE', token });
  assert.equal(deleted.status, 200);
  const emptyList = await jsonApi('/api/virtual-teachers');
  assert.deepEqual(emptyList.data.characters, []);
  const deletedImage = await fetch(`${baseUrl}${created.character.imagePath}`);
  assert.equal(deletedImage.status, 404);
  const deletedModel = await fetch(`${baseUrl}${created.character.modelPath}`);
  assert.equal(deletedModel.status, 404);
});
