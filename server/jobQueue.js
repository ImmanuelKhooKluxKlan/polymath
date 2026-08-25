const {
  SQSClient,
  SendMessageCommand,
  ReceiveMessageCommand,
  DeleteMessageCommand,
} = require('@aws-sdk/client-sqs');

class JobQueue {
  constructor({ queueUrl, region }) {
    this.queueUrl = String(queueUrl || '').trim();
    this.enabled = Boolean(this.queueUrl);
    this.client = this.enabled ? new SQSClient({ region: String(region || 'us-east-2') }) : null;
    this.running = false;
  }

  async enqueue(job) {
    if (!this.enabled) return false;
    await this.client.send(new SendMessageCommand({
      QueueUrl: this.queueUrl,
      MessageBody: JSON.stringify(job),
    }));
    return true;
  }

  async start(handler) {
    if (!this.enabled || this.running) return;
    this.running = true;
    while (this.running) {
      try {
        const response = await this.client.send(new ReceiveMessageCommand({
          QueueUrl: this.queueUrl,
          MaxNumberOfMessages: 1,
          WaitTimeSeconds: 20,
          VisibilityTimeout: 43200,
        }));
        for (const message of response.Messages || []) {
          try {
            const job = JSON.parse(message.Body || '{}');
            await handler(job);
            await this.client.send(new DeleteMessageCommand({
              QueueUrl: this.queueUrl,
              ReceiptHandle: message.ReceiptHandle,
            }));
          } catch (error) {
            console.error('Queued job failed and will be retried:', error);
          }
        }
      } catch (error) {
        console.error('Job queue poll failed:', error);
        await new Promise((resolve) => setTimeout(resolve, 5000));
      }
    }
  }

  stop() {
    this.running = false;
  }
}

function createJobQueue(options) {
  return new JobQueue(options);
}

module.exports = { createJobQueue };
