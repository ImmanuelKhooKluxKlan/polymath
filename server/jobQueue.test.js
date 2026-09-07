const assert = require('node:assert/strict');
const test = require('node:test');
const { JobQueue, boundedInteger } = require('./jobQueue');

function commandName(command) {
  return command?.constructor?.name || '';
}

test('successful queue work is acknowledged only after the handler finishes', async () => {
  const commands = [];
  const client = { send: async (command) => { commands.push(command); return {}; } };
  const queue = new JobQueue({ queueUrl: 'queue-url', client });
  queue.running = true;

  let handled = false;
  await queue.handleMessage({ Body: '{"type":"media-transcription","jobId":"job-1"}', ReceiptHandle: 'receipt-1' }, async (job) => {
    assert.equal(job.jobId, 'job-1');
    handled = true;
  });

  assert.equal(handled, true);
  assert.deepEqual(commands.map(commandName), ['DeleteMessageCommand']);
});

test('failed queue work becomes visible for retry and is not deleted', async () => {
  const commands = [];
  const client = { send: async (command) => { commands.push(command); return {}; } };
  const queue = new JobQueue({ queueUrl: 'queue-url', client, retryDelaySeconds: 17 });
  queue.running = true;
  const originalError = console.error;
  console.error = () => {};
  try {
    await queue.handleMessage({ Body: '{"jobId":"job-2"}', ReceiptHandle: 'receipt-2' }, async () => {
      throw new Error('temporary failure');
    });
  } finally {
    console.error = originalError;
  }

  assert.deepEqual(commands.map(commandName), ['ChangeMessageVisibilityCommand']);
  assert.equal(commands[0].input.VisibilityTimeout, 17);
});

test('queue safety settings are bounded', () => {
  const queue = new JobQueue({
    queueUrl: 'queue-url',
    client: { send: async () => ({}) },
    concurrency: 500,
    visibilityTimeoutSeconds: 10,
    visibilityHeartbeatSeconds: 9999,
  });
  assert.equal(queue.concurrency, 10);
  assert.equal(queue.visibilityTimeoutSeconds, 60);
  assert.equal(queue.visibilityHeartbeatSeconds, 50);
  assert.equal(boundedInteger('bad', 4, 1, 10), 4);
});
