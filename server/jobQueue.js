const {
  ChangeMessageVisibilityCommand,
  SQSClient,
  SendMessageCommand,
  ReceiveMessageCommand,
  DeleteMessageCommand,
} = require('@aws-sdk/client-sqs');

function boundedInteger(value, fallback, minimum, maximum) {
  const parsed = Math.floor(Number(value));
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(minimum, Math.min(maximum, parsed));
}

class JobQueue {
  constructor({
    queueUrl,
    region,
    concurrency = 1,
    visibilityTimeoutSeconds = 300,
    visibilityHeartbeatSeconds = 60,
    retryDelaySeconds = 30,
    client = null,
  }) {
    this.queueUrl = String(queueUrl || '').trim();
    this.enabled = Boolean(this.queueUrl);
    this.client = this.enabled ? (client || new SQSClient({ region: String(region || 'us-east-2') })) : null;
    this.concurrency = boundedInteger(concurrency, 1, 1, 10);
    this.visibilityTimeoutSeconds = boundedInteger(visibilityTimeoutSeconds, 300, 60, 43200);
    this.visibilityHeartbeatSeconds = Math.min(
      this.visibilityTimeoutSeconds - 10,
      boundedInteger(visibilityHeartbeatSeconds, 60, 15, 3600),
    );
    this.retryDelaySeconds = boundedInteger(retryDelaySeconds, 30, 1, 900);
    this.running = false;
    this.workers = [];
  }

  async enqueue(job) {
    if (!this.enabled) return false;
    await this.client.send(new SendMessageCommand({
      QueueUrl: this.queueUrl,
      MessageBody: JSON.stringify(job),
    }));
    return true;
  }

  async changeVisibility(message, seconds) {
    if (!message?.ReceiptHandle) return;
    await this.client.send(new ChangeMessageVisibilityCommand({
      QueueUrl: this.queueUrl,
      ReceiptHandle: message.ReceiptHandle,
      VisibilityTimeout: seconds,
    }));
  }

  async handleMessage(message, handler) {
    let visibilityRenewalRunning = false;
    const heartbeat = setInterval(async () => {
      if (visibilityRenewalRunning || !this.running) return;
      visibilityRenewalRunning = true;
      try {
        await this.changeVisibility(message, this.visibilityTimeoutSeconds);
      } catch (error) {
        console.error('Job queue visibility heartbeat failed:', error);
      } finally {
        visibilityRenewalRunning = false;
      }
    }, this.visibilityHeartbeatSeconds * 1000);
    heartbeat.unref?.();

    try {
      const job = JSON.parse(message.Body || '{}');
      await handler(job);
      await this.client.send(new DeleteMessageCommand({
        QueueUrl: this.queueUrl,
        ReceiptHandle: message.ReceiptHandle,
      }));
    } catch (error) {
      console.error('Queued job failed and will be retried:', error);
      try {
        await this.changeVisibility(message, this.retryDelaySeconds);
      } catch (visibilityError) {
        console.error('Queued job retry scheduling failed:', visibilityError);
      }
    } finally {
      clearInterval(heartbeat);
    }
  }

  async consume(handler) {
    while (this.running) {
      try {
        const response = await this.client.send(new ReceiveMessageCommand({
          QueueUrl: this.queueUrl,
          MaxNumberOfMessages: 1,
          WaitTimeSeconds: 20,
          VisibilityTimeout: this.visibilityTimeoutSeconds,
        }));
        for (const message of response.Messages || []) {
          await this.handleMessage(message, handler);
        }
      } catch (error) {
        console.error('Job queue poll failed:', error);
        await new Promise((resolve) => setTimeout(resolve, 5000));
      }
    }
  }

  async start(handler) {
    if (!this.enabled || this.running) return;
    this.running = true;
    this.workers = Array.from({ length: this.concurrency }, () => this.consume(handler));
    await Promise.all(this.workers);
  }

  stop() {
    this.running = false;
  }
}

function createJobQueue(options) {
  return new JobQueue(options);
}

module.exports = { JobQueue, boundedInteger, createJobQueue };
