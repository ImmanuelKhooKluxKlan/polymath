const path = require('node:path');
const dotenv = require('dotenv');

dotenv.config({ path: path.join(__dirname, '.env'), quiet: true });

function required(name) {
  const value = String(process.env[name] || '').trim();
  if (!value) throw new Error(`${name} is missing from server/.env`);
  return value;
}

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function main() {
  const endpointId = required('RUNPOD_SERVERLESS_ENDPOINT_ID');
  const apiKey = required('RUNPOD_API_KEY');
  const baseUrl = `https://api.runpod.ai/v2/${endpointId}`;
  const headers = {
    Authorization: `Bearer ${apiKey}`,
    'Content-Type': 'application/json',
  };
  const request = async (pathname, options = {}) => {
    const response = await fetch(`${baseUrl}${pathname}`, { ...options, headers });
    const text = await response.text();
    const payload = text ? JSON.parse(text) : {};
    if (!response.ok) {
      throw new Error(`RunPod HTTP ${response.status}: ${payload.error || payload.message || text}`);
    }
    return payload;
  };

  const submitted = await request('/run', {
    method: 'POST',
    body: JSON.stringify({
      input: {
        action: 'bootstrap_model_copies',
        version: String(process.env.MUSCRIPTOR_TEST_VERSION || 'v001').trim(),
      },
      policy: {
        executionTimeout: 60 * 60 * 1000,
        ttl: 2 * 60 * 60 * 1000,
      },
    }),
  });
  if (!submitted.id) throw new Error('RunPod did not return a bootstrap job ID');

  console.log(`Bootstrap job submitted: ${submitted.id}`);
  const deadline = Date.now() + 60 * 60 * 1000;
  let lastMessage = '';
  while (Date.now() < deadline) {
    const status = await request(`/status/${encodeURIComponent(submitted.id)}`);
    const message = `${status.status || 'UNKNOWN'} ${status.output?.progress || ''}`.trim();
    if (message !== lastMessage) {
      console.log(message);
      lastMessage = message;
    }
    if (status.status === 'COMPLETED') {
      const result = status.output || {};
      if (!result.testerWeightsPath || Number(result.weightsBytes) < 5_000_000_000) {
        throw new Error('Bootstrap completed without a verified Polymath Large checkpoint');
      }
      console.log(`Tester ready: ${result.testerWeightsPath}`);
      console.log(`Weights verified: ${result.weightsBytes} bytes`);
      return;
    }
    if (['CANCELLED', 'FAILED', 'TIMED_OUT'].includes(status.status)) {
      throw new Error(status.output?.error || `Bootstrap ended with ${status.status}`);
    }
    await sleep(2_000);
  }
  throw new Error('Bootstrap exceeded the one-hour local wait limit');
}

main().catch((error) => {
  console.error(`Polymath bootstrap failed: ${error.message}`);
  process.exitCode = 1;
});
