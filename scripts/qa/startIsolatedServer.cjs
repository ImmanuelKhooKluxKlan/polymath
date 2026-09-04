'use strict';

const path = require('node:path');

const dataRoot = path.resolve(process.argv[2] || path.join(process.cwd(), '.qa-data'));
process.env.PORT = String(Number(process.argv[3]) || 3010);
process.env.POLYMATH_DATA_DIR = dataRoot;
process.env.NODE_ENV = 'test';
process.env.ADMIN_EMAILS = 'visual-qa@polymath.test,visual-mobile@polymath.test';
process.env.REGISTRATION_OTP_TEST_CODE = '123456';
process.env.MUSCRIPTOR_ENABLED = 'false';
process.env.RUNPOD_CHAT_BOSS_ENDPOINT_ID = 'visual-only';
process.env.RUNPOD_API_KEY = 'visual-only';

const { app } = require('../../server/server');

app.listen(Number(process.env.PORT), '127.0.0.1', () => {
  process.stdout.write(`Isolated Polymath QA server listening on ${process.env.PORT}\n`);
});
